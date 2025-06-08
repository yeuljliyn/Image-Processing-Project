import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import qrcode

VIDEO_DIR = "videos"
use_webcam = False
video_list = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
current_video_name = video_list[0] if video_list else None
video_path = os.path.join(VIDEO_DIR, current_video_name) if current_video_name else None
cap = cv2.VideoCapture(video_path)
model_path = "scrfd_10g_bnkps.onnx"

resize_width, resize_height = 854, 480
output_width, output_height = 854, 480

cap = cv2.VideoCapture(video_path) if video_path else None
detector = SCRFD(model_path)
detector.prepare(ctx_id=-1)

fps = cap.get(cv2.CAP_PROP_FPS) if cap else 30
fps = 30 if fps == 0 or not isinstance(fps, (int, float)) or np.isnan(fps) else fps

filter_enabled = True
auto_scroll_enabled = False

current_face_idx = None
total_faces = 0
target_center = None
indexed_faces = []
frame_count = 0
zoom = 150

face_counter = {}
known_faces = {}
next_id = 0

sepia_kernel = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])
# ───────── Tkinter GUI 구성 ─────────
root = tk.Tk()
root.title("Face Zoom Viewer")
root.geometry(f"{output_width}x{output_height+200}")

canvas = tk.Canvas(root, width=output_width, height=output_height)
canvas.pack()

label = tk.Label(root, text="Focused Face [0/0]", font=("Helvetica", 14))
label.pack()

frame_image = None
image_id = canvas.create_image(0, 0, anchor=tk.NW, image=None)

zoom_var = tk.IntVar(value=zoom)
tk.Scale(root, from_=0, to=300, orient=tk.HORIZONTAL, label="Zoom", variable=zoom_var).pack()

source_frame = tk.Frame(root)
source_frame.pack(pady=5)

source_label = tk.Label(root, text=f"🎞 Source: {os.path.basename(video_path)}", font=("Helvetica", 10))
source_label.pack()

def draw_faces_with_index(frame, faces):
    for idx, (x, y, w, h, cx, cy) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

video_var = tk.StringVar(value=current_video_name)
video_dropdown = ttk.Combobox(root, textvariable=video_var, values=video_list, state="readonly", width=50)
video_dropdown.pack(pady=4)

def change_video(event=None):
    global cap, frame_count, current_face_idx, target_center, indexed_faces, known_faces, face_counter
    selected = video_var.get()
    cap.release()
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, selected))
    frame_count = 0
    current_face_idx = None
    target_center = None
    indexed_faces = []
    known_faces = {}
    face_counter = {}
    print(f"[🎞 재생 영상 변경됨] → {selected}")

video_dropdown.bind("<<ComboboxSelected>>", change_video)

def switch_to_video():
    global cap, use_webcam
    if cap.isOpened():
        cap.release()
    use_webcam = False
    cap = cv2.VideoCapture(video_path)
    source_label.config(text=f"🎞 Source: {os.path.basename(video_path)}")

def update_label():
    if current_face_idx is not None:
        label.config(text=f"Focused Face [{current_face_idx+1}/{total_faces}]")
    else:
        label.config(text=f"Select a face [0/{total_faces}]")

def toggle_filter():
    global filter_enabled
    filter_enabled = not filter_enabled
    toggle_btn.config(text="Filter: ON" if filter_enabled else "Filter: OFF")

def toggle_auto_scroll():
    global auto_scroll_enabled
    auto_scroll_enabled = not auto_scroll_enabled
    auto_scroll_btn.config(text="Auto Scroll: ON" if auto_scroll_enabled else "Auto Scroll: OFF")
    auto_scroll_status_label.config(
        text="✅ Auto Scroll ON" if auto_scroll_enabled else "❌ Auto Scroll OFF",
        fg="green" if auto_scroll_enabled else "red")

def prev_face():
    global current_face_idx, target_center
    if total_faces > 0 and current_face_idx is not None:
        current_face_idx = (current_face_idx - 1 + total_faces) % total_faces
        target_center = indexed_faces[current_face_idx][4:6]
        update_label()

def next_face():
    global current_face_idx, target_center
    if total_faces > 0 and current_face_idx is not None:
        current_face_idx = (current_face_idx + 1) % total_faces
        target_center = indexed_faces[current_face_idx][4:6]
        update_label()

def show_qr_statistics():
    if not face_counter:
        messagebox.showinfo("통계 없음", "얼굴 통계 데이터가 아직 없습니다.")
        return

    summary = '\n'.join([f"Face ID {fid}: {cnt} frames" for fid, cnt in face_counter.items()])
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(summary)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")

    top = tk.Toplevel(root)
    top.title("📊 얼굴 통계 QR")
    qr_img = ImageTk.PhotoImage(img)
    tk.Label(top, image=qr_img).pack()
    top.qr_img = qr_img  # 참조 유지

def switch_to_webcam():
    global cap, use_webcam
    if cap.isOpened():
        cap.release()
    use_webcam = True
    cap = cv2.VideoCapture(0)
    source_label.config(text="🎥 Source: Webcam")

# ✅ 그 다음 버튼 생성

tk.Button(source_frame, text="Use Webcam", command=switch_to_webcam).pack(side=tk.LEFT, padx=5)
tk.Button(source_frame, text="Use Video", command=switch_to_video).pack(side=tk.LEFT, padx=5)

btn_frame = tk.Frame(root)
btn_frame.pack()

auto_scroll_status_label = tk.Label(root, text="❌ Auto Scroll OFF", font=("Helvetica", 12), fg="red")
auto_scroll_status_label.pack(pady=4)

tk.Button(btn_frame, text="◀ Prev", width=10, command=prev_face).pack(side=tk.LEFT, padx=5)
auto_scroll_btn = tk.Button(btn_frame, text="Auto Scroll: OFF", width=14, command=toggle_auto_scroll)
auto_scroll_btn.pack(side=tk.LEFT, padx=5)
toggle_btn = tk.Button(btn_frame, text="Filter: ON", width=12, command=toggle_filter)
toggle_btn.pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Next ▶", width=10, command=next_face).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="📊 QR 통계 보기", width=16, command=show_qr_statistics).pack(side=tk.LEFT, padx=5)
def match_faces(prev_faces, new_faces):
    matched = [None] * len(new_faces)
    used = set()
    for i, nf in enumerate(new_faces):
        cx, cy = nf[4], nf[5]
        min_dist = float('inf')
        min_idx = -1
        for j, pf in enumerate(prev_faces):
            if j in used:
                continue
            px, py = pf[4], pf[5]
            dist = (px - cx)**2 + (py - cy)**2
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        if min_idx != -1:
            used.add(min_idx)
        matched[i] = nf
    return matched

def update():
    global indexed_faces, total_faces, frame_image, frame_count

    ret, frame = cap.read()
    if not ret:
        root.after(33, update)
        return
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return

    frame_count += 1
    original_frame = frame.copy()
    orig_h, orig_w, _ = original_frame.shape

    # 얼굴 검출 (5프레임마다)
    if frame_count % 5 == 0:
        detect_frame = cv2.resize(original_frame, (resize_width, resize_height))
        dets = detector.detect(detect_frame, input_size=(640, 640))[0]

        scale_x = orig_w / resize_width
        scale_y = orig_h / resize_height
        restored_dets = []
        for det in dets:
            x1, y1, x2, y2, score = det[:5]
            restored_dets.append([
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
                score
            ])
        new_faces = [(x1, y1, x2 - x1, y2 - y1, (x1 + x2) // 2, (y1 + y2) // 2)
                     for x1, y1, x2, y2, _ in restored_dets]

        if not indexed_faces:
            indexed_faces = new_faces
        else:
            indexed_faces = match_faces(indexed_faces, new_faces)

        total_faces = len(indexed_faces)
        update_label()

    zoom_value = zoom_var.get()

    if current_face_idx is None:
        display_frame = original_frame.copy()
        draw_faces_with_index(display_frame, indexed_faces)
        display_frame = cv2.resize(display_frame, (output_width, output_height))
        frame_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
        canvas.itemconfig(image_id, image=frame_image)
        root.after(int(1000 / fps), update)
        return

    # 특정 얼굴만 줌인해서 보기
    if current_face_idx is not None and total_faces > 0 and target_center is not None:
        def get_nearest_face(faces, target_center):
            tx, ty = target_center
            return min(faces, key=lambda f: (f[4] - tx)**2 + (f[5] - ty)**2)

        target_face = get_nearest_face(indexed_faces, target_center)
        if target_face:
            x, y, bw, bh, cx, cy = target_face
            pad = zoom_value

            w_half = (bw + 2 * pad) // 2
            h_half = (bh + 2 * pad) // 2
            x1 = max(cx - w_half, 0)
            y1 = max(cy - h_half, 0)
            x2 = min(cx + w_half, orig_w)
            y2 = min(cy + h_half, orig_h)

            focused = original_frame[y1:y2, x1:x2].copy()

            draw_faces_with_index(focused, [
                (f[0] - x1, f[1] - y1, f[2], f[3], f[4] - x1, f[5] - y1)
                for f in indexed_faces if x1 <= f[0] <= x2 and y1 <= f[1] <= y2
            ])

            focused = cv2.UMat(focused)
            focused = cv2.resize(focused, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)

            if filter_enabled:
                focused = cv2.bilateralFilter(focused, d=9, sigmaColor=75, sigmaSpace=75)
                focused = cv2.detailEnhance(focused, sigma_s=15, sigma_r=0.075)
                sharpen_kernel = np.array([[0, -1, 0],
                                           [-1, 5.2, -1],
                                           [0, -1, 0]])
                focused = cv2.filter2D(focused, -1, sharpen_kernel)

            focused = focused.get()
            frame_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(focused, cv2.COLOR_BGR2RGB)))
            canvas.itemconfig(image_id, image=frame_image)

    root.after(int(1000 / fps), update)

def key_event(event):
    global current_face_idx, target_center
    if event.char.isdigit():
        idx = int(event.char)
        if idx < total_faces:
            current_face_idx = idx
            target_center = indexed_faces[current_face_idx][4:6]
            update_label()

root.bind("<Key>", key_event)
update()
root.mainloop()

if cap:
    cap.release()
cv2.destroyAllWindows()
