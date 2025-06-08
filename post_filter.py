import cv2
import numpy as np

cap = cv2.VideoCapture("videos/asepa.mp4")
frame_id = 0

# ✅ Sepia 행렬 정의
sepia_kernel = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])

while True:
    ret, frame = cap.read()
    if not ret or frame_id > 100:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # ✅ OpenCV는 float32 형식을 선호
    sepia = cv2.transform(frame.astype(np.float32), sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Edges", edges)
    cv2.imshow("Sepia", sepia)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
