import cv2
import numpy as np
import qrcode
from insightface.model_zoo.scrfd import SCRFD
from collections import defaultdict

# 모델 및 영상
cap = cv2.VideoCapture("videos/asepa.mp4")
model = SCRFD("scrfd_10g_bnkps.onnx")
model.prepare(ctx_id=-1)

face_counter = defaultdict(int)
known_faces = {}
next_id = 0
frame_id = 0

def get_face_id(cx, cy, threshold=50):
    for fid, (px, py) in known_faces.items():
        if (cx - px)**2 + (cy - py)**2 < threshold**2:
            return fid
    known_faces[next_id] = (cx, cy)
    global next_id
    next_id += 1
    return next_id - 1

# 등장 통계 계산
while True:
    ret, frame = cap.read()
    if not ret or frame_id > 300:
        break
    dets = model.detect(frame, input_size=(640, 640))[0]
    for det in dets:
        x1, y1, x2, y2, _ = det[:5]
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
        fid = get_face_id(cx, cy)
        face_counter[fid] += 1
    frame_id += 1

cap.release()

# 통계 텍스트 만들기
summary = "얼굴 등장 통계 (상위 5):\n"
for fid, count in sorted(face_counter.items(), key=lambda x: -x[1])[:5]:
    summary += f"ID {fid} → {count}회\n"

print(summary)

# ✅ QR 코드 생성
qr = qrcode.QRCode(version=1, box_size=10, border=4)
qr.add_data(summary)
qr.make(fit=True)
img_qr = qr.make_image(fill='black', back_color='white')
img_qr.save("output/face_summary_qr.png")

# ✅ OpenCV로 띄우기
img_qr_cv = cv2.imread("output/face_summary_qr.png")
cv2.imshow("📊 얼굴 등장 통계 QR", img_qr_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
