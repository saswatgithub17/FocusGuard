# 🎯 FocusGuard – Study Concentration Detector

**FocusGuard** is a real-time face-tracking system that helps users stay focused while studying or working.  
Using **OpenCV** and **MediaPipe**, it continuously monitors your face through the webcam and alerts you with a **beep sound** whenever you get distracted or move out of frame.

---

## 🧠 How It Works

1. The webcam captures live video feed.  
2. MediaPipe’s **Face Detection** model identifies your face position.  
3. If your face is not detected for more than **2 seconds**, FocusGuard assumes you’re distracted.  
4. A **red indicator** appears on screen, and a **beep sound** plays as an alert.  
5. When you’re focused (face detected), the indicator turns **green** again.

---

## 🖥️ Features

✅ Real-time face tracking using **MediaPipe**  
✅ Visual feedback (🟢 Green = Focused, 🔴 Red = Distracted)  
✅ Audible alert using **winsound**  
✅ Lightweight and easy to use  
✅ Fully customizable alert delay and beep tone  

---

## ⚙️ Requirements

Install dependencies using:
pip install opencv-python mediapipe
