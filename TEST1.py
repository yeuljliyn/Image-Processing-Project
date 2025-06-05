import cv2
import mediapipe as mp
import numpy as np

# 설정
video_path = "뉴진스.mp4"
output_path = "테스트.mp4"
cap = cv2.VideoCapture(video_path)

mp_face_detection = mp.solutions.face_detection
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

# FPS 기반 재생 속도 조절
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# 슬라이더 설정
cv2.namedWindow("Focused Face")
cv2.createTrackbar("Zoom", "Focused Face", 50, 150, lambda x: None)  # 여백 조절

target_center = None  # 선택된 얼굴 중심 좌표

def draw_faces_with_index(frame, detections, w, h):
    faces = []
    for idx, detection in enumerate(detections):
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        cx, cy = x + bw // 2, y + bh // 2
        faces.append((x, y, bw, bh, cx, cy))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, f'{idx}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return faces

def get_nearest_face(faces, target_center):
    tx, ty = target_center
    min_dist = float('inf')
    best_face = None
    for face in faces:
        _, _, _, _, cx, cy = face
        dist = (tx - cx) ** 2 + (ty - cy) ** 2
        if dist < min_dist:
            min_dist = dist
            best_face = face
    return best_face

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            faces = draw_faces_with_index(frame, results.detections, w, h)

            if target_center is None:
                cv2.imshow("Select a Face (Press index number)", frame)
                key = cv2.waitKey(0)
                if ord('0') <= key <= ord(str(min(len(faces) - 1, 9))):
                    idx = int(chr(key))
                    _, _, _, _, cx, cy = faces[idx]
                    target_center = (cx, cy)
                    continue
                else:
                    print("잘못된 입력입니다.")
                    continue

            target_face = get_nearest_face(faces, target_center)
            if target_face:
                x, y, bw, bh, _, _ = target_face
                pad = cv2.getTrackbarPos("Zoom", "Focused Face")

                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + bw + pad, w)
                y2 = min(y + bh + pad, h)

                focused = frame[y1:y2, x1:x2]
                fh, fw = focused.shape[:2]
                output_frame = np.zeros((h, w, 3), dtype=np.uint8)
                start_y = (h - fh) // 2
                start_x = (w - fw) // 2
                output_frame[start_y:start_y+fh, start_x:start_x+fw] = focused

                if out is None:
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

                out.write(output_frame)
                cv2.imshow("Focused Face", output_frame)
            else:
                cv2.imshow("Focused Face", frame)
        else:
            cv2.imshow("Focused Face", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

