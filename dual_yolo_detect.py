import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Parse user input arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='Primary YOLO model (e.g., yolo11s.pt or custom best.pt)')
parser.add_argument('--model2', default=None,
                    help='Secondary YOLO model (optional)')
parser.add_argument('--source', required=True,
                    help='Image, folder, video, or usb0 / picamera0')
parser.add_argument('--thresh', default=0.5,
                    help='Confidence threshold')
parser.add_argument('--resolution', default=None,
                    help='WxH resolution (e.g., 640x480)')
parser.add_argument('--record', action='store_true',
                    help='Record output video (requires resolution)')
args = parser.parse_args()

# ----------------------------
# Validate model paths
# ----------------------------
if not os.path.exists(args.model):
    print("ERROR: Primary model not found.")
    sys.exit(0)

if args.model2 and not os.path.exists(args.model2):
    print("ERROR: Secondary model not found.")
    sys.exit(0)

# ----------------------------
# Load YOLO models
# ----------------------------
model1 = YOLO(args.model, task='detect')
labels1 = model1.names

model2 = None
labels2 = None
if args.model2:
    model2 = YOLO(args.model2, task='detect')
    labels2 = model2.names

print("Loaded primary model:", args.model)
if model2:
    print("Loaded secondary model:", args.model2)

# ----------------------------
# Parse source type
# ----------------------------
img_source = args.source
img_ext_list = ['.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f"Unsupported file extension: {ext}")
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print("Invalid source input.")
    sys.exit(0)

# ----------------------------
# Parse resolution
# ----------------------------
resize = False
if args.resolution:
    resize = True
    resW, resH = map(int, args.resolution.split('x'))

# ----------------------------
# Setup recording
# ----------------------------
record = args.record
if record:
    if source_type not in ['video', 'usb']:
        print("Recording only works for video or USB camera.")
        sys.exit(0)
    if not resize:
        print("Recording requires --resolution.")
        sys.exit(0)

    recorder = cv2.VideoWriter(
        'demo1.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH)
    )

# ----------------------------
# Load source
# ----------------------------
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*')
                 if os.path.splitext(f)[1] in img_ext_list]

elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(
        main={"format": 'XRGB8888', "size": (resW, resH)}
    ))
    cap.start()

# ----------------------------
# Colors for each model
# ----------------------------
color1 = (0, 255, 0)       # green
color2 = (0, 128, 255)     # orange/blue

# ----------------------------
# Inference loop
# ----------------------------
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

while True:
    t_start = time.perf_counter()

    # ----------------------------
    # Load frame
    # ----------------------------
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print("All images processed.")
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera disconnected.")
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # ----------------------------
    # Resize if needed
    # ----------------------------
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # ----------------------------
    # Run inference on both models
    # ----------------------------
    results1 = model1(frame, verbose=False)
    dets1 = results1[0].boxes

    dets2 = []
    if model2:
        results2 = model2(frame, verbose=False)
        dets2 = results2[0].boxes

    # ----------------------------
    # Combine detections
    # ----------------------------
    all_dets = [('m1', d) for d in dets1]
    if model2:
        all_dets += [('m2', d) for d in dets2]

    object_count = 0

    # ----------------------------
    # Draw detections
    # ----------------------------
    for src, det in all_dets:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(det.cls.item())
        conf = det.conf.item()

        if conf < float(args.thresh):
            continue

        if src == 'm1':
            classname = labels1[classidx]
            color = color1
        else:
            classname = labels2[classidx]
            color = color2

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{classname}: {int(conf*100)}%"
        cv2.putText(frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        object_count += 1

    # ----------------------------
    # FPS display
    # ----------------------------
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f"FPS: {avg_frame_rate:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

    cv2.putText(frame, f"Objects: {object_count}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    cv2.imshow("Dual YOLO Detection", frame)
    if record:
        recorder.write(frame)

    # ----------------------------
    # Key controls
    # ----------------------------
    key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite("capture.png", frame)

    # ----------------------------
    # FPS calculation
    # ----------------------------
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ----------------------------
# Cleanup
# ----------------------------
print(f"Average FPS: {avg_frame_rate:.2f}")
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
