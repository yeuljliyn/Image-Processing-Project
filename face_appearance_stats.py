# face_appearance_visual.py

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD

cap = cv2.VideoCapture("videos/asepa.mp4")
model = SCRFD("scrfd_10g_bnkps.onnx")
model.prepare(ctx_id=-1)

face_counter = {}
known_faces = {}
next_id = 0
frame_id = 0

def assign_id(cx, cy):
    global next_id
    for fid, (px, py) in known_faces.items():
        if (cx - px)**2 + (cy - py)**2 < 50**2:
            return fid
    known_faces[next_id] = (cx, cy)
    next_id += 1
    return next_id - 1

while True:
    ret, frame = cap.read()
    if not ret or frame_id > 300:
        break

    dets = model.detect(frame, input_size=(640, 640))[0]
    for det in dets:
        x1, y1, x2, y2, _ = det[:5]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        fid = assign_id(cx, cy)
        face_counter[fid] = face_counter.get(fid, 0) + 1

        # 시각화
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{fid} ({face_counter[fid]})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Face Stats Viewer", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
