import os
import sys
import argparse
import glob
import time
import webbrowser

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# GLOBAL LIST FOR CLICKABLE BOXES
# -----------------------------
clicked_boxes = []   # (xmin, ymin, xmax, ymax, classname)

# -----------------------------
# MOUSE CALLBACK FUNCTION
# -----------------------------
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for (xmin, ymin, xmax, ymax, classname) in clicked_boxes:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                query = classname.replace(" ", "+")
                url = f"https://www.google.com/search?q={query}"
                webbrowser.open(url)
                print(f"Searching Google for: {classname}")
                break


# -----------------------------
# ARGUMENT PARSER
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(model_path):
    print("ERROR: Model path invalid.")
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

# -----------------------------
# DETERMINE SOURCE TYPE
# -----------------------------
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
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
    print("Invalid source.")
    sys.exit(0)

# -----------------------------
# RESOLUTION PARSING
# -----------------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# -----------------------------
# RECORDING SETUP
# -----------------------------
if record:
    if source_type not in ['video','usb']:
        print("Recording only works for video or USB camera.")
        sys.exit(0)
    if not user_res:
        print("Specify --resolution to record.")
        sys.exit(0)

    recorder = cv2.VideoWriter(
        'demo1.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH)
    )

# -----------------------------
# LOAD SOURCE
# -----------------------------
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]

elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# -----------------------------
# COLORS & FPS
# -----------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# -----------------------------
# SETUP WINDOW + MOUSE CALLBACK
# -----------------------------
cv2.namedWindow('YOLO detection results')
cv2.setMouseCallback('YOLO detection results', mouse_callback)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    t_start = time.perf_counter()

    # Load frame
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

    # Resize
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    clicked_boxes.clear()   # reset clickable list

    # Draw detections
    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        classidx = int(det.cls.item())
        classname = labels[classidx]
        conf = det.conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f"{classname}: {int(conf*100)}%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                          (xmin+labelSize[0], label_ymin+baseLine-10),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            object_count += 1

            # ADD TO CLICKABLE LIST
            clicked_boxes.append((xmin, ymin, xmax, ymax, classname))

    # FPS
    if source_type in ['video','usb','picamera']:
        cv2.putText(frame, f"FPS: {avg_frame_rate:.2f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f"Objects: {object_count}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5 if source_type in ['video','usb','picamera'] else 0)

    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    # FPS calc
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps)

    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f"Average pipeline FPS: {avg_frame_rate:.2f}")
if source_type in ['video','usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
