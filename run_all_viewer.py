# run_all_viewer.py
import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD

cap = cv2.VideoCapture("videos/asepa.mp4")
model = SCRFD("scrfd_10g_bnkps.onnx")
model.prepare(ctx_id=-1)

sepia_kernel = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])

face_counter = {}
known_faces = {}
next_id = 0
frame_id = 0
display_width, display_height = 426, 240

def assign_id(cx, cy, threshold=50):
    global next_id
    for fid, (px, py) in known_faces.items():
        if (cx - px)**2 + (cy - py)**2 < threshold**2:
            return fid
    known_faces[next_id] = (cx, cy)
    next_id += 1
    return next_id - 1

while True:
    ret, frame = cap.read()
    if not ret or frame_id > 300:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    sepia = cv2.transform(frame.astype(np.float32), sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    vis = frame.copy()
    dets = model.detect(frame, input_size=(640, 640))[0]
    for det in dets:
        x1, y1, x2, y2, _ = det[:5]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        fid = assign_id(cx, cy)
        face_counter[fid] = face_counter.get(fid, 0) + 1
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(vis, f"ID:{fid} ({face_counter[fid]})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

    # 리사이즈
    stacked = np.vstack([
        np.hstack([
            cv2.resize(to_bgr(frame), (display_width, display_height)),
            cv2.resize(to_bgr(gray), (display_width, display_height)),
            cv2.resize(to_bgr(edges), (display_width, display_height))
        ]),
        np.hstack([
            cv2.resize(to_bgr(sepia), (display_width, display_height)),
            cv2.resize(vis, (display_width, display_height)),
            np.zeros((display_height, display_width, 3), dtype=np.uint8)
        ])
    ])

    cv2.imshow("📊 필터 + 통계 시각화 (by YOU)", stacked)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
