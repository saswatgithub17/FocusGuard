import cv2
import mediapipe as mp
import time
import threading
import csv
import os
import numpy as np
import winsound
from datetime import datetime

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ── Color Palette (BGR) ───────────────────────────────────────────────────────
CLR_GREEN       = (60, 220, 100)
CLR_AMBER       = (0, 180, 255)
CLR_RED         = (40, 40, 220)
CLR_WHITE       = (240, 240, 240)
CLR_DARK        = (15, 12, 22)
CLR_PANEL_BG    = (25, 18, 38)
CLR_BORDER_FOC  = (60, 200, 80)
CLR_BORDER_DIS  = (40, 40, 200)
CLR_TEXT_DIM    = (140, 130, 160)

# ── Layout ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H    = 1280, 720
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO       = cv2.FONT_HERSHEY_PLAIN

# ── Tunable Constants ─────────────────────────────────────────────────────────
ALERT_DELAY         = 2.0    # seconds before first alert
ALERT_COOLDOWN      = 3.0    # seconds between beeps
ESCALATE_2          = 5.0    # seconds for louder beep
ESCALATE_3          = 10.0   # seconds for screen flash
POMODORO_FOCUS_SEC  = 25 * 60
POMODORO_BREAK_SEC  = 5 * 60
YAW_THRESHOLD       = 0.28   # head turned sideways limit (ratio)
PITCH_THRESHOLD     = 0.22   # head tilted down limit (ratio)

# ── FaceMesh landmark indices ─────────────────────────────────────────────────
# Nose tip, left/right temple, chin — used for head pose estimation
NOSE_TIP    = 1
LEFT_EAR    = 234
RIGHT_EAR   = 454
CHIN        = 152
FOREHEAD    = 10


# ── Session State ─────────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.start_time         = time.time()
        self.focused_seconds    = 0.0
        self.distracted_seconds = 0.0
        self.distraction_count  = 0
        self.distraction_log    = []   # list of (timestamp, duration)

        self._last_state        = "focused"
        self._state_start       = time.time()

        # Alert state
        self.last_seen_time     = time.time()
        self.last_beep_time     = 0.0
        self.is_beeping         = False   # thread lock flag
        self.distraction_start  = None

        # Pomodoro
        self.pomodoro_start     = time.time()
        self.pomodoro_phase     = "focus"   # "focus" | "break"

        # Flash overlay
        self.flash_alpha        = 0.0
        self.flash_active       = False

    def update(self, is_focused: bool):
        now = time.time()
        dt  = now - self._state_start
        self._state_start = now

        if self._last_state == "focused":
            self.focused_seconds += dt
        else:
            self.distracted_seconds += dt

        if is_focused and self._last_state == "distracted":
            # Returned to focus — log distraction
            dur = now - (self.distraction_start or now)
            self.distraction_log.append((datetime.now().strftime("%H:%M:%S"), round(dur, 1)))
            self.distraction_start = None

        if not is_focused and self._last_state == "focused":
            self.distraction_count += 1
            self.distraction_start = now

        self._last_state = "focused" if is_focused else "distracted"

    @property
    def focus_score(self) -> int:
        total = self.focused_seconds + self.distracted_seconds
        if total < 1:
            return 100
        return int(100 * self.focused_seconds / total)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def pomodoro_remaining(self) -> tuple:
        """Returns (phase, seconds_remaining)."""
        phase_dur = POMODORO_FOCUS_SEC if self.pomodoro_phase == "focus" else POMODORO_BREAK_SEC
        elapsed   = time.time() - self.pomodoro_start
        remaining = phase_dur - elapsed
        if remaining <= 0:
            # Switch phase
            self.pomodoro_phase = "break" if self.pomodoro_phase == "focus" else "focus"
            self.pomodoro_start = time.time()
            remaining = phase_dur
        return self.pomodoro_phase, max(0, remaining)

    def save_log(self, path="focus_log.csv"):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["session_date", "session_duration_s", "focus_score_pct",
                        "distraction_count", "focused_s", "distracted_s"])
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"),
                        round(self.elapsed), self.focus_score,
                        self.distraction_count,
                        round(self.focused_seconds), round(self.distracted_seconds)])
            w.writerow([])
            w.writerow(["distraction_time", "duration_s"])
            for ts, dur in self.distraction_log:
                w.writerow([ts, dur])
        print(f"[FocusGuard] Session log saved → {os.path.abspath(path)}")


# ── Head Pose Estimation ──────────────────────────────────────────────────────
def estimate_gaze(landmarks, iw, ih):
    """
    Returns (yaw_ratio, pitch_ratio).
    yaw_ratio  > YAW_THRESHOLD   → head turned sideways
    pitch_ratio > PITCH_THRESHOLD → head tilted down
    """
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y])

    nose     = pt(NOSE_TIP)
    left_ear = pt(LEFT_EAR)
    right_ear= pt(RIGHT_EAR)
    chin     = pt(CHIN)
    forehead = pt(FOREHEAD)

    face_width  = np.linalg.norm(right_ear - left_ear) + 1e-6
    face_height = np.linalg.norm(chin - forehead) + 1e-6

    mid_x   = (left_ear[0] + right_ear[0]) / 2
    yaw     = abs(nose[0] - mid_x) / face_width

    mid_y   = (forehead[1] + chin[1]) / 2
    pitch   = (nose[1] - mid_y) / face_height

    return yaw, pitch


# ── Alert System (non-blocking) ───────────────────────────────────────────────
def _beep_async(freq, dur_ms):
    winsound.Beep(freq, dur_ms)

def trigger_beep(freq=900, dur_ms=400):
    t = threading.Thread(target=_beep_async, args=(freq, dur_ms), daemon=True)
    t.start()


# ── Drawing Helpers ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, x, y, w, h, r, color, alpha=1.0, thickness=-1):
    if alpha < 1.0:
        overlay = img.copy()
        _draw_rr(overlay, x, y, w, h, r, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        _draw_rr(img, x, y, w, h, r, color, thickness)

def _draw_rr(img, x, y, w, h, r, color, thickness=-1):
    if thickness == -1:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, -1)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, -1)
        for cx, cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(img, (cx,cy), r, color, -1)
    else:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, thickness)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, thickness)
        for cx, cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(img, (cx,cy), r, color, thickness)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def draw_hud(img, session: Session, status: str, distracted_dur: float,
             yaw: float, pitch: float, flash_alpha: float):
    ih, iw = img.shape[:2]

    # ── Screen flash overlay ──
    if flash_alpha > 0:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (iw, ih), (0, 0, 180), -1)
        cv2.addWeighted(overlay, flash_alpha, img, 1 - flash_alpha, 0, img)

    is_focused   = status == "focused"
    status_color = CLR_GREEN if is_focused else CLR_RED
    phase, pomo_rem = session.pomodoro_remaining()
    in_break = phase == "break"

    # ── Top HUD bar ──────────────────────────────────────────────────────────
    bar_h = 70
    draw_rounded_rect(img, 0, 0, iw, bar_h, 0, CLR_PANEL_BG, alpha=0.88)
    cv2.line(img, (0, bar_h), (iw, bar_h), status_color, 2)

    # Status pill
    pill_x, pill_y = 16, 14
    pill_w, pill_h = 160, 42
    draw_rounded_rect(img, pill_x, pill_y, pill_w, pill_h, 10, status_color, alpha=0.25)
    draw_rounded_rect(img, pill_x, pill_y, pill_w, pill_h, 10, status_color, thickness=1)
    status_label = "FOCUSED" if is_focused else "DISTRACTED"
    (tw, th), _ = cv2.getTextSize(status_label, FONT, 0.65, 2)
    cv2.putText(img, status_label,
                (pill_x + (pill_w - tw)//2, pill_y + (pill_h + th)//2 - 2),
                FONT, 0.65, status_color, 2, cv2.LINE_AA)

    # Focus score bar
    score = session.focus_score
    score_x = 200
    cv2.putText(img, "FOCUS", (score_x, 28), FONT, 0.45, CLR_TEXT_DIM, 1, cv2.LINE_AA)
    bar_track_w = 220
    cv2.rectangle(img, (score_x, 35), (score_x + bar_track_w, 52), (50, 40, 70), -1)
    fill_w = int(bar_track_w * score / 100)
    score_color = CLR_GREEN if score >= 70 else (CLR_AMBER if score >= 40 else CLR_RED)
    cv2.rectangle(img, (score_x, 35), (score_x + fill_w, 52), score_color, -1)
    cv2.putText(img, f"{score}%", (score_x + bar_track_w + 8, 52),
                FONT, 0.6, score_color, 1, cv2.LINE_AA)

    # Session time
    t_x = 480
    cv2.putText(img, "SESSION", (t_x, 28), FONT, 0.45, CLR_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, fmt_time(session.elapsed), (t_x, 58),
                FONT, 0.9, CLR_WHITE, 2, cv2.LINE_AA)

    # Distractions count
    d_x = 620
    cv2.putText(img, "DISTRACTIONS", (d_x, 28), FONT, 0.45, CLR_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, str(session.distraction_count), (d_x + 30, 58),
                FONT, 0.9, CLR_WHITE, 2, cv2.LINE_AA)

    # Pomodoro
    p_x = 760
    phase_label = "BREAK" if in_break else "POMODORO"
    phase_color = CLR_AMBER if in_break else CLR_GREEN
    cv2.putText(img, phase_label, (p_x, 28), FONT, 0.45, phase_color, 1, cv2.LINE_AA)
    cv2.putText(img, fmt_time(pomo_rem), (p_x, 58),
                FONT, 0.9, phase_color, 2, cv2.LINE_AA)

    # Head pose indicators
    h_x = 940
    cv2.putText(img, "HEAD POSE", (h_x, 28), FONT, 0.45, CLR_TEXT_DIM, 1, cv2.LINE_AA)
    yaw_color   = CLR_RED if yaw   > YAW_THRESHOLD   else CLR_GREEN
    pitch_color = CLR_RED if pitch > PITCH_THRESHOLD  else CLR_GREEN
    cv2.putText(img, f"YAW {yaw:.2f}", (h_x, 48), FONT, 0.45, yaw_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"PITCH {pitch:.2f}", (h_x, 64), FONT, 0.45, pitch_color, 1, cv2.LINE_AA)

    # FPS placeholder area (top-right)
    # drawn by main loop

    # ── Bottom stats strip ────────────────────────────────────────────────────
    strip_y = ih - 44
    draw_rounded_rect(img, 0, strip_y, iw, 44, 0, CLR_PANEL_BG, alpha=0.82)
    cv2.line(img, (0, strip_y), (iw, strip_y), (60, 50, 80), 1)

    focused_pct   = int(100 * session.focused_seconds    / max(session.elapsed, 1))
    distract_pct  = int(100 * session.distracted_seconds / max(session.elapsed, 1))
    cv2.putText(img, f"Focused: {fmt_time(session.focused_seconds)}  ({focused_pct}%)",
                (16, strip_y + 28), FONT, 0.52, CLR_GREEN, 1, cv2.LINE_AA)
    cv2.putText(img, f"Distracted: {fmt_time(session.distracted_seconds)}  ({distract_pct}%)",
                (360, strip_y + 28), FONT, 0.52, CLR_RED, 1, cv2.LINE_AA)

    if not is_focused and distracted_dur > 0:
        cv2.putText(img, f"Away for: {distracted_dur:.1f}s",
                    (750, strip_y + 28), FONT, 0.52, CLR_AMBER, 1, cv2.LINE_AA)

    if in_break:
        cv2.putText(img, "  BREAK TIME — alerts paused",
                    (950, strip_y + 28), FONT, 0.52, CLR_AMBER, 1, cv2.LINE_AA)

    # ── Escalation warning banner ─────────────────────────────────────────────
    if not is_focused and distracted_dur >= ESCALATE_2 and not in_break:
        warn_y = bar_h + 10
        draw_rounded_rect(img, 16, warn_y, 460, 36, 8, (0, 0, 160), alpha=0.85)
        level = "!!! RETURN TO FOCUS !!!" if distracted_dur >= ESCALATE_3 else "!! Please refocus !!"
        cv2.putText(img, level, (26, warn_y + 24),
                    FONT, 0.65, (80, 80, 255), 2, cv2.LINE_AA)


def draw_fps(img, fps: float):
    iw = img.shape[1]
    cv2.putText(img, f"FPS {fps:.0f}", (iw - 90, 28),
                FONT, 0.5, CLR_TEXT_DIM, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

session     = Session()
prev_time   = time.time()
flash_alpha = 0.0

WIN_NAME = "FocusGuard — Study Concentration Detector"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("[FocusGuard] Running — press Q to quit and save log.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (CAM_W, CAM_H))
    now = time.time()

    # FPS
    fps       = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    # ── Process frame ─────────────────────────────────────────────────────────
    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    yaw, pitch       = 0.0, 0.0
    face_detected    = False
    looking_away     = False

    if results.multi_face_landmarks:
        face_detected = True
        lms = results.multi_face_landmarks[0].landmark
        yaw, pitch = estimate_gaze(lms, CAM_W, CAM_H)
        looking_away = (yaw > YAW_THRESHOLD) or (pitch > PITCH_THRESHOLD)

        # Draw subtle mesh dots on face
        ih, iw = img.shape[:2]
        for i in [NOSE_TIP, LEFT_EAR, RIGHT_EAR, CHIN, FOREHEAD]:
            lm = lms[i]
            cx, cy = int(lm.x * iw), int(lm.y * ih)
            cv2.circle(img, (cx, cy), 4, CLR_GREEN, -1, cv2.LINE_AA)

    is_focused = face_detected and not looking_away

    # ── Pomodoro break → suppress alerts ─────────────────────────────────────
    phase, pomo_rem = session.pomodoro_remaining()
    in_break = phase == "break"

    # ── Update session tracking ───────────────────────────────────────────────
    session.update(is_focused or in_break)

    if is_focused or in_break:
        session.last_seen_time = now
        flash_alpha = max(0.0, flash_alpha - 0.08)

    # ── Distraction duration ──────────────────────────────────────────────────
    distracted_dur = 0.0
    if not is_focused and not in_break:
        distracted_dur = now - session.last_seen_time

    # ── Escalating alert logic ────────────────────────────────────────────────
    if not is_focused and not in_break and distracted_dur > ALERT_DELAY:
        cooldown_ok = (now - session.last_beep_time) > ALERT_COOLDOWN

        if distracted_dur >= ESCALATE_3:
            # Level 3: screen flash + aggressive beep
            flash_alpha = min(flash_alpha + 0.15, 0.45)
            if cooldown_ok:
                trigger_beep(freq=1200, dur_ms=600)
                session.last_beep_time = now
        elif distracted_dur >= ESCALATE_2:
            # Level 2: louder, faster beep
            flash_alpha = max(0.0, flash_alpha - 0.03)
            if cooldown_ok:
                trigger_beep(freq=1000, dur_ms=500)
                session.last_beep_time = now
        else:
            # Level 1: gentle nudge
            flash_alpha = max(0.0, flash_alpha - 0.05)
            if cooldown_ok:
                trigger_beep(freq=750, dur_ms=300)
                session.last_beep_time = now
    else:
        flash_alpha = max(0.0, flash_alpha - 0.06)

    # ── Draw HUD ──────────────────────────────────────────────────────────────
    status = "focused" if (is_focused or in_break) else "distracted"
    draw_hud(img, session, status, distracted_dur, yaw, pitch, flash_alpha)
    draw_fps(img, fps)

    cv2.imshow(WIN_NAME, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
session.save_log()
print(f"[FocusGuard] Session complete. Focus score: {session.focus_score}%")