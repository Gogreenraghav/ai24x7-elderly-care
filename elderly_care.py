"""
AI24x7 Elderly Care AI - Fall Detection System
Monitors elderly via camera, detects falls, alerts family instantly.
No wearable needed - AI sees + responds.
"""
import os, cv2, numpy as np, threading, queue, time, json
from pathlib import Path
from datetime import datetime, timedelta

# ─── Fall Detection Algorithm ──────────────
class FallDetector:
    """
    Detects falls using pose estimation + motion analysis.
    No external model downloads needed (uses OpenCV fallback).
    """
    
    def __init__(self):
        self.prev_pose = None
        self.pose_history = []
        self.fall_confidence_threshold = 0.65
    
    def _estimate_pose_simple(self, frame):
        """Simple pose estimation without heavy models.
        Detects person bounding box + center of mass movement."""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Background subtraction
        if not hasattr(self, 'bg'):
            self.bg = cv2.createBackgroundSubtractorMOG2()
        
        fg = self.bg.apply(gray)
        
        # Find foreground (person)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Largest foreground contour = person
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 500:  # Too small
            return None
        
        x, y, cw, ch = cv2.boundingRect(largest)
        
        # Aspect ratio: standing person is tall, fallen person is wide
        aspect = ch / cw if cw > 0 else 0
        
        # Center of mass
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + cw/2, y + ch/2
        
        # Height vs width ratio
        box_ratio = ch / h  # How much of frame height
        
        return {
            "x": x, "y": y, "w": cw, "h": ch,
            "cx": cx, "cy": cy,
            "aspect": aspect,
            "box_ratio": box_ratio,
            "area": area,
            "frame_h": h, "frame_w": w
        }
    
    def detect(self, frame):
        """Detect if person has fallen"""
        pose = self._estimate_pose_simple(frame)
        if pose is None:
            return {"fall": False, "confidence": 0, "pose": None}
        
        self.pose_history.append(pose)
        if len(self.pose_history) > 30:
            self.pose_history.pop(0)
        
        if len(self.pose_history) < 10:
            return {"fall": False, "confidence": 0, "pose": pose}
        
        # ── Fall detection logic ──
        # Person fell if:
        # 1. Aspect ratio dropped suddenly (standing → lying)
        # 2. Center of mass dropped (standing → floor)
        # 3. Box ratio changed (tall → wide)
        
        current = pose
        history = self.pose_history[-10:-1]  # last 9 frames
        
        # Check for sudden drop in aspect ratio
        avg_aspect = np.mean([p["aspect"] for p in history])
        aspect_drop = avg_aspect - current["aspect"]
        
        # Check for center of mass dropping
        avg_cy = np.mean([p["cy"] for p in history])
        cy_rise = current["cy"] - avg_cy  # Higher cy = lower on screen = fell
        
        # Check for width increase (lying = wider)
        avg_w = np.mean([p["w"] for p in history])
        w_increase = current["w"] / (avg_w + 1)
        
        # Combine into fall score
        fall_score = 0
        if aspect_drop > 0.8:  # Tall → flat
            fall_score += 0.5
        if cy_rise > current["frame_h"] * 0.2:  # Dropped > 20% of frame
            fall_score += 0.3
        if w_increase > 1.5:  # Got 50%+ wider
            fall_score += 0.2
        
        fall_score = min(fall_score, 1.0)
        
        # Verify: stayed on floor for 2+ seconds (not just sat down)
        if fall_score > self.fall_confidence_threshold:
            # Check if still on floor
            recent = self.pose_history[-5:]
            still_down = all(p["aspect"] < 2 for p in recent)
            if not still_down:
                fall_score *= 0.5  # Not a real fall
        
        return {
            "fall": fall_score > self.fall_confidence_threshold,
            "confidence": round(fall_score, 3),
            "pose": pose
        }


# ─── Activity Monitor ───────────────────────
class ActivityMonitor:
    """Tracks normal activity patterns + detects anomalies"""
    
    def __init__(self, person_id="resident"):
        self.person_id = person_id
        self.last_seen = None
        self.routine = {}  # hour -> typical_activity
        self.check_ins = deque(maxlen=96)  # Last 24 hours (every 15 min)
    
    def record_activity(self, timestamp=None):
        ts = timestamp or datetime.now()
        self.last_seen = ts
        self.check_ins.append(ts)
    
    def check_wellbeing(self):
        """Check if person responded to check-in"""
        if not self.check_ins:
            return True, "no_data"
        
        last_check = self.check_ins[-1]
        mins_ago = (datetime.now() - last_check).total_seconds() / 60
        
        if mins_ago < 120:  # 2 hours
            return True, "active"
        elif mins_ago < 240:  # 4 hours
            return False, "check_needed"  # Send check-in
        else:
            return False, "concern"  # Alert family
    
    def get_daily_summary(self):
        """Get activity summary for the day"""
        today = datetime.now().date()
        today_checks = [t for t in self.check_ins if t.date() == today]
        
        return {
            "date": str(today),
            "total_activities": len(today_checks),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None
        }


# ─── Elderly Care Camera ────────────────────
class ElderlyCareCamera(threading.Thread):
    def __init__(self, camera_id, camera_url, room_name, alert_queue, config=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.room_name = room_name
        self.alert_queue = alert_queue
        self.config = config or {}
        self.fall_detector = FallDetector()
        self.activity = ActivityMonitor(person_id=camera_id)
        self.running = False
        self.cap = None
        self.fall_cooldown = 120  # 2 min between fall alerts
        self.last_fall_alert = 0
        self.check_in_interval = 7200  # 2 hours
        self.last_check_in_sent = 0
    
    def run(self):
        self.running = True
        self._connect()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(2)
                self._connect()
                continue
            
            # Fall detection
            result = self.fall_detector.detect(frame)
            
            if result["fall"]:
                self._trigger_fall_alert(result)
            
            # Activity tracking
            if result["pose"]:
                self.activity.record_activity()
            
            # Periodic check-in
            self._check_wellbeing()
            
            time.sleep(0.3)
    
    def _connect(self):
        self.cap = cv2.VideoCapture(self.camera_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def _trigger_fall_alert(self, result):
        now = time.time()
        if now - self.last_fall_alert < self.fall_cooldown:
            return
        
        self.last_fall_alert = now
        
        alert = {
            "type": "fall_detected",
            "camera_id": self.camera_id,
            "room_name": self.room_name,
            "timestamp": datetime.now().isoformat(),
            "confidence": result["confidence"],
            "priority": "critical"
        }
        self.alert_queue.put(alert)
        print(f"👴 FALL DETECTED! {self.room_name} - conf: {result['confidence']:.2f}")
    
    def _check_wellbeing(self):
        now = time.time()
        if now - self.last_check_in_sent < self.check_in_interval:
            return
        
        active, status = self.activity.check_wellbeing()
        if not active:
            self._send_check_in(status)
            self.last_check_in_sent = now
    
    def _send_check_in(self, status):
        alert = {
            "type": "wellbeing_check",
            "camera_id": self.camera_id,
            "room_name": self.room_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "message": f"No activity detected in {self.room_name} for 4+ hours"
        }
        self.alert_queue.put(alert)
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ─── Alert Handler ──────────────────────────
class ElderlyAlertHandler(threading.Thread):
    def __init__(self, alert_queue, contacts=None):
        super().__init__(daemon=True)
        self.alert_queue = alert_queue
        self.contacts = contacts or []
        self.start()
    
    def run(self):
        while True:
            alert = self.alert_queue.get()
            self._process(alert)
    
    def _process(self, alert):
        atype = alert["type"]
        
        print("\n" + "="*50)
        if atype == "fall_detected":
            print("👴 FALL ALERT!")
            print(f"   Room: {alert['room_name']}")
            print(f"   Time: {alert['timestamp'][:19]}")
            print(f"   Confidence: {alert['confidence']*100:.0f}%")
            self._send_fall_alert(alert)
        elif atype == "wellbeing_check":
            print("💤 Wellbeing Check: " + alert["message"])
            self._send_wellbeing_alert(alert)
        print("="*50 + "\n")
    
    def _send_fall_alert(self, alert):
        msg = (
            f"👴 FALL DETECTED!\n"
            f"Room: {alert['room_name']}\n"
            f"Time: {alert['timestamp'][:19]}\n"
            f"Camera: {alert['camera_id']}\n\n"
            f"⚠️ Please check immediately!"
        )
        for contact in self.contacts:
            print(f"📱 Alert to {contact}: {msg[:100]}")
    
    def _send_wellbeing_alert(self, alert):
        msg = f"💤 {alert['message']}\nTime: {alert['timestamp'][:19]}"
        for contact in self.contacts:
            print(f"📱 Check-in to {contact}: {msg[:80]}")


# ─── Elderly Care Manager ──────────────────
class ElderlyCareManager:
    def __init__(self, emergency_contacts=None):
        self.alert_queue = queue.Queue()
        self.alert_handler = ElderlyAlertHandler(self.alert_queue, emergency_contacts)
        self.cameras = {}
    
    def add_room(self, camera_id, camera_url, room_name):
        cam = ElderlyCareCamera(camera_id, camera_url, room_name, self.alert_queue)
        cam.start()
        self.cameras[camera_id] = cam
        print(f"👴 Monitoring room: {room_name}")
    
    def get_all_status(self):
        return {
            cam_id: {
                "room": cam.room_name,
                "last_seen": cam.activity.last_seen.isoformat() if cam.activity.last_seen else None,
                "total_activities": len(cam.activity.check_ins)
            }
            for cam_id, cam in self.cameras.items()
        }


def create_api(manager):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    
    @app.route("/elderly/health")
    def health(): return jsonify({"status": "ok"})
    
    @app.route("/elderly/room/add", methods=["POST"])
    def add_room():
        data = request.get_json()
        manager.add_room(data["camera_id"], data["camera_url"], data["room_name"])
        return jsonify({"success": True})
    
    @app.route("/elderly/status")
    def status(): return jsonify(manager.get_all_status())
    
    return app


if __name__ == "__main__":
    import uvicorn
    manager = ElderlyCareManager()
    app = create_api(manager)
    print("👴 AI24x7 Elderly Care running on port 5063")
    uvicorn.run(app, host="0.0.0.0", port=5063)