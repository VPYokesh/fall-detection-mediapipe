# Fall Detection Using Human Pose Estimation

This project implements an automated **fall detection system** using **MediaPipe Pose** and **OpenCV**.  
It analyzes human body posture from video streams to detect fall events based on pose orientation, body height reduction, and temporal consistency.

The system supports **batch processing of multiple videos** and logs all detected fall events into a **single consolidated log file**.

---

## 📌 Objective

To detect fall events in video data by analyzing human body posture using pose estimation techniques.

---

## 🚀 Features

- Human pose estimation using MediaPipe Pose
- Extraction of key body landmarks (head, shoulders, hips, knees, ankles)
- Posture classification:
  - Standing
  - Bent
  - Fall Detected (Lying)
- Fall detection using:
  - Torso orientation (shoulder–hip angle)
  - Reduction in vertical body height
  - Temporal confirmation across consecutive frames
- Batch processing of multiple input videos
- Annotated output videos with pose skeleton overlay
- Centralized logging of fall events with timestamps

---

## 📂 Project Structure

fall-detection-mediapipe/
│
├── pose_detection.py
├── requirements.txt
├── detection_report.txt
├── fall_detection_log.txt
├── README.md
│
├── videos/
│ ├── video1.mp4
│ ├── video2.mp4
│
├── annotated_outputs/
│ ├── annotated_video1.mp4
│ ├── annotated_video2.mp4


> Note: The virtual environment folder is intentionally not included.

---

## 🛠️ Requirements

- Python **3.10**
- MediaPipe **0.10.14**
- OpenCV
- NumPy
- Protobuf

Install all required dependencies using:

```bash
pip install -r requirements.txt

```

▶️ How to Run

Place all input video files inside the videos/ folder.

Run the fall detection script:

python pose_detection.py

📤 Output

annotated_outputs/
Contains output videos with skeleton overlays and posture labels.

fall_detection_log.txt
Stores detected fall events with timestamps and corresponding video names.

Example log entry:

Fall detected at 00:00:42 in video1.mp4
