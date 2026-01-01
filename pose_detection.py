import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import timedelta

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "annotated_outputs"
LOG_FILE = "fall_detection_log.txt"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def torso_angle(shoulder_mid, hip_mid):
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return np.degrees(np.arctan2(abs(dy), abs(dx)))

log_file = open(LOG_FILE, "w")

for video_name in os.listdir(VIDEO_FOLDER):
    if not video_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_FOLDER, video_name)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(
        OUTPUT_FOLDER, f"annotated_{video_name}"
    )

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0
    fall_frames = 0
    FALL_CONFIRM_FRAMES = 4
    fall_logged = False

    log_file.write(f"\nProcessing Video: {video_name}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        posture = "Unknown"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            head_y = lm[0].y
            ankle_y = max(lm[27].y, lm[28].y)

            shoulder_mid = [
                (lm[11].x + lm[12].x) / 2,
                (lm[11].y + lm[12].y) / 2
            ]
            hip_mid = [
                (lm[23].x + lm[24].x) / 2,
                (lm[23].y + lm[24].y) / 2
            ]

            height_ratio = abs(ankle_y - head_y)
            angle = torso_angle(shoulder_mid, hip_mid)

            is_lying = (
                angle < 45 and
                height_ratio < 0.5
            )

            if is_lying:
                posture = "Fall Detected"
                fall_frames += 1
            elif angle < 70:
                posture = "Bent"
                fall_frames = 0
                fall_logged = False
            else:
                posture = "Standing"
                fall_frames = 0
                fall_logged = False

            if fall_frames >= FALL_CONFIRM_FRAMES and not fall_logged:
                time_sec = frame_count // fps
                timestamp = str(timedelta(seconds=time_sec))
                log_file.write(
                    f"Fall detected at {timestamp} in {video_name}\n"
                )
                log_file.flush()
                fall_logged = True

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(
            frame,
            posture,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out.write(frame)

    cap.release()
    out.release()

log_file.close()
cv2.destroyAllWindows()
