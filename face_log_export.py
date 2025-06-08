# face_logger_gui.py

import cv2
from insightface.model_zoo.scrfd import SCRFD
import tkinter as tk
from tkinter import scrolledtext

cap = cv2.VideoCapture("videos/asepa.mp4")
model = SCRFD("scrfd_10g_bnkps.onnx")
model.prepare(ctx_id=-1)

fps = cap.get(cv2.CAP_PROP_FPS)

# Tkinter 윈도우 설정
root = tk.Tk()
root.title("얼굴 인식 로그 뷰어")
log_area = scrolledtext.ScrolledText(root, width=80, height=25)
log_area.pack()

frame_id = 0

def update():
    global frame_id
    ret, frame = cap.read()
    if not ret or frame_id > 300:
        return

    dets = model.detect(frame, input_size=(640, 640))[0]
    timestamp = frame_id / fps

    for i, det in enumerate(dets):
        x1, y1, x2, y2, _ = det[:5]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        log = f"[{frame_id:04}] {timestamp:.2f}s - 얼굴 {i} 좌표: ({cx}, {cy})"
        log_area.insert(tk.END, log + "\n")
        log_area.see(tk.END)

    frame_id += 1
    root.after(30, update)

update()
root.mainloop()
cap.release()
