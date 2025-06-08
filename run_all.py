# run_all.py

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD

# ─────────────────────────────
# 설정
video_path = "videos/asepa.mp4"
model_path = "scrfd_10g_bnkps.onnx"
max_frames = 300
display_width = 426
display_height = 240

# ─────────────────────────────
# 초기화
cap = cv2.VideoCapture(video_path)
model = SCRFD(model_path)
model.prepare(ctx_id=-1)

face_counter = {}
known_faces = {}
next_id = 0
frame_id = 0

sepia_kernel = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])

def assign_id(cx, cy, threshold=50):
    global next_id
    for fid, (px, py) in known_faces.items():
        if (cx - px)**2 + (cy - py)**2 < threshold**2:
            return fid
    known_faces[next_id] = (cx, cy)
    next_id += 1
    return next_id - 1

# ─────────────────────────────
# 루프
while True:
    ret, frame = cap.read()
    if not ret or frame_id > max_frames:
        break

    # 필터 처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    sepia = cv2.transform(frame.astype(np.float32), sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    # 얼굴 인식 및 표시
    vis_frame = frame.copy()
    dets = model.detect(frame, input_size=(640, 640))[0]

    for det in dets:
        x1, y1, x2, y2, _ = det[:5]
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
        fid = assign_id(cx, cy)
        face_counter[fid] = face_counter.get(fid, 0) + 1

        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"ID:{fid} ({face_counter[fid]})",
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    # 리사이즈
    orig = cv2.resize(frame, (display_width, display_height))
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray_colored = cv2.resize(gray_colored, (display_width, display_height))
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.resize(edges_colored, (display_width, display_height))
    sepia = cv2.resize(sepia, (display_width, display_height))
    vis_frame = cv2.resize(vis_frame, (display_width, display_height))

    # 하나의 창에 합치기 (2행 x 3열)
    row1 = np.hstack([orig, gray_colored, edges_colored])
    row2 = np.hstack([sepia, vis_frame, np.zeros_like(orig)])
    combined = np.vstack([row1, row2])

    cv2.imshow("🧼 필터 + 🧠 얼굴통계 + 📊 로그 통합 보기", combined)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
