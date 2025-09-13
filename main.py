# Built-in imports
import sys

# Third party imports
import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

# Local import
from tracker import ObjectTracker
from cachetools import TTLCache

# Logger configuration
logger.remove()
logger.add(sys.stdout, level="TRACE")

MAX_NUMBER_OF_PREDICTIONS = 30
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.6

# PolygonZone
# Video1 (source: video1.avi)
POLY_ZONE_VIDEO1 = np.array(
    [
        [1, 349],
        [11, 344],
        [25, 351],
        [413, 171],
        [409, 225],
        [638, 255],
        [639, 420],
        [0, 418],
    ]
)
# Video2 (source: video2.avi)
POLY_ZONE_VIDEO2 = np.array(
    [
        [-1, 61],
        [6, 54],
        [46, 62],
        [77, 54],
        [85, 36],
        [123, 0],
        [507, 2],
        [481, 477],
        [2, 475],
    ]
)
# Video3 (source: video3.avi)
POLY_ZONE_VIDEO3 = np.array(
    [
        [1, 59],
        [45, 62],
        [68, 59],
        [81, 47],
        [79, 35],
        [119, 0],
        [509, 0],
        [498, 478],
        [2, 475],
    ]
)
# Video4 (source: video4.avi)
POLY_ZONE_VIDEO4 = np.array(
    [
        [-1, 61],
        [5, 55],
        [42, 60],
        [74, 55],
        [79, 40],
        [121, 0],
        [509, 0],
        [499, 479],
        [3, 477],
    ]
)
# Video5 (source: video5.avi)
POLY_ZONE_VIDEO5 = np.array(
    [
        [0, 144],
        [40, 143],
        [177, 100],
        [209, 100],
        [227, 93],
        [486, 211],
        [487, 240],
        [509, 264],
        [564, 258],
        [709, 327],
        [707, 356],
        [607, 478],
        [443, 471],
        [357, 434],
        [248, 446],
        [1, 378],
    ]
)

# Video6 (source: video6-cutted.mp4)
POLY_ZONE_VIDEO6 = np.array(
    [
        [5, 171],
        [97, 192],
        [177, 196],
        [195, 205],
        [218, 208],
        [391, 370],
        [378, 396],
        [388, 416],
        [412, 429],
        [433, 432],
        [438, 445],
        [69, 449],
        [4, 397],
    ]
)

# Video7 (source: video7-cutted.mp4)
POLY_ZONE_VIDEO7 = np.array([[1, 148], [485, 101], [711, 219], [711, 475], [1, 475]])

# Video8 (source: video8-cutted.mp4)
POLY_ZONE_VIDEO8 = np.array(
    [[2, 148], [493, 104], [709, 221], [709, 477], [1, 474], [1, 474]]
)

# Video9 (source: video9.avi)
POLY_ZONE_VIDEO9 = np.array(
    [[0, 150], [0, 150], [467, 99], [709, 222], [713, 475], [1, 476]]
)

# Video10 (source: video10.avi)
POLY_ZONE_VIDEO10 = np.array(
    [[0, 151], [471, 101], [709, 234], [710, 473], [23, 473], [1, 420]]
)


def main():
    # Load classification YOLO model
    cls_model = YOLO("cls-model.pt")
    # Initialize Tracker
    tracker = ObjectTracker()

    # Prepare path and polygon zone
    dq = TTLCache(maxsize=100, ttl=2)  # 2-second TTL
    video_choosen = "video6"  # You can change this to video1, video2, ..., video10
    poly_used = POLY_ZONE_VIDEO6  # Change this accordingly to the video chosen
    file_path = (
        "videos//video6-cutted.mp4"  # Change this accordingly to the video chosen
    )
    still_bg_path = (
        "videos//video6-cutted.png"  # Change this accordingly to the video chosen
    )

    # Load Video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {file_path}")
        sys.exit(1)

    # Get video properties (width, height, fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20  # fallback if 0

    # Define codec and create VideoWriter object for each transformation
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Options: 'XVID', 'MJPG', 'MP4V', 'X264'
    out_frame_main = cv2.VideoWriter(
        f"{video_choosen}/output_frame_main.avi",
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    out_frame_diff = cv2.VideoWriter(
        f"{video_choosen}/output_frame_diff.avi",
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    out_frame_edge = cv2.VideoWriter(
        f"{video_choosen}/output_frame_edge.avi",
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    out_frame_cleaned_edge = cv2.VideoWriter(
        f"{video_choosen}/output_frame_cleaned_edge.avi",
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    # 1. Load still background (as being ground truth frame)
    firstframe = cv2.imread(still_bg_path)
    firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    firstframe_blur = cv2.GaussianBlur(firstframe_gray, (3, 3), 0)
    cv2.imshow("First frame", firstframe_blur)
    cv2.imwrite(f"{video_choosen}/firstframe.png", firstframe_blur)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream or cannot fetch the frame.")
            break
        frame_height, frame_width, _ = frame.shape

        # 2. Convert frame to gray scale and apply Gaussian blur
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)

        # 3. Find difference between current frame and first frame (the ground truth frame)
        frame_diff = cv2.absdiff(firstframe_blur, frame_blur)
        cv2.imshow("frame diff", frame_diff)
        out_frame_diff.write(cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR))

        # 4. Automatic Canny Edge Detection using the computed median
        median = np.median(frame_blur)
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        # Canny Edge Detection
        edged = cv2.Canny(frame_diff, lower, upper, apertureSize=3, L2gradient=True)
        cv2.imshow("CannyEdgeDet", edged)
        out_frame_edge.write(cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR))

        # 5. Morphological operations to remove noise - Dilation followed by Erosion (Closing)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, k2, iterations=2)
        cv2.imshow("Cleaned Edges", thresh)
        out_frame_cleaned_edge.write(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

        # 6. Find contours of all detected objects
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 7. Get bounding box for each contour
        detections = []
        count = 0
        for c in cnts:
            contourArea = cv2.contourArea(c)
            if contourArea > 50 and contourArea < 10000:
                (x, y, w, h) = cv2.boundingRect(c)

                # extra sanity filters (optional)
                aspect = w / float(h) if h > 0 else 0
                if 0.1 < aspect < 10:
                    detections.append([x, y, w, h])
                    count += 1
        ids, abandoned_objects = tracker.update(frame, detections)

        # 8. Loop over abandoned objects and classify them
        for objects in abandoned_objects:
            id, x, y, w, h, _ = objects
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Check if the center of the bounding box is inside the polygon zone
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check if the center point is inside the polygon zone
            is_inside = (
                cv2.pointPolygonTest(poly_used, (center_x, center_y), False) >= 0
            )
            if not is_inside:
                continue
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

            # YOLO object classification
            cropped = frame[y1:y2, x1:x2]
            result = cls_model.predict(source=cropped, imgsz=256, conf=0.6, device=0)[0]
            probs = result.probs
            top1 = probs.top1
            top1_conf = probs.top1conf
            pred_object = cls_model.names[top1]

            if id not in dq:
                dq[id] = []
            logger.debug(f"id: {id}, dq[id]: {dq[id]}")
            if len(dq[id]) == MAX_NUMBER_OF_PREDICTIONS:
                most_common = max(set(dq[id]), key=dq[id].count)
                cv2.putText(
                    frame,
                    f"{most_common}",
                    (x2, y2 - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif top1_conf > CLASSIFICATION_CONFIDENCE_THRESHOLD:
                dq[id].append(pred_object)

        # 9. Draw the polygon zone on the frame + show the frame
        cv2.polylines(frame, [poly_used], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imshow("main", frame)
        out_frame_main.write(frame)

        if cv2.waitKey(15) == ord("q"):
            break

    out_frame_main.release()
    out_frame_diff.release()
    out_frame_edge.release()
    out_frame_cleaned_edge.release()


if __name__ == "__main__":
    main()
