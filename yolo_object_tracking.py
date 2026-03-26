import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("yolov8n.pt")

# Videos list
video_list = [
    "videos/1sample.mp4",
    "videos/2sample.mp4"
]

for idx, video_path in enumerate(video_list):

    print(f"Processing Video {idx+1}...")

    cap = cv2.VideoCapture(video_path)

    # Fixed output size
    frame_width = 800
    frame_height = 600
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = f"output/result_{idx+1}.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )

    tracker = DeepSort(max_age=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Video {idx+1} finished")
            break

        # Resize frame
        frame = cv2.resize(frame, (800, 600))

        # Detection
        results = model.predict(frame, conf=0.5)[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w = x2 - x1
            h = y2 - y1

            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append(([x1, y1, w, h], conf, class_name))

        # Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            class_name = track.get_det_class()

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{class_name} ID: {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Show video
        cv2.imshow(f"Video {idx+1}", frame)

        # Save video
        out.write(frame)

        # Handle key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Skipping to next video...")
            

    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("All videos processed successfully!")