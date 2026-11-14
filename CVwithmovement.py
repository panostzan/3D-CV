
import cv2
import numpy as np

# ============================================================
# BLACK OBJECT TRACKING + OPTICAL FLOW
#
# Uses:
# 1. Brightness thresholding to detect black objects.
# 2. Shi-Tomasi to detect corners inside those objects.
# 3. Lucas-Kanade optical flow to track corner movement.
#
# Q = Quit
# R = Reset tracking
# C = Clear trails
# ============================================================


# STEP 1: SETTINGS

DARK_THRESHOLD = 65       # Maximum brightness considered black
MIN_OBJECT_AREA = 500     # Ignore tiny dark regions
MAX_CORNERS = 100         # Maximum tracked features
DETECT_INTERVAL = 10      # Detect new features every 10 frames


# STEP 2: CREATE A MASK FOR DARK OBJECTS

def create_dark_mask(frame):

    # Convert image to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dark pixels become white in the mask.
    # Everything brighter becomes black.
    mask = cv2.inRange(gray, 0, DARK_THRESHOLD)

    # Remove small isolated noise.
    kernel = np.ones((3, 3), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    # Find connected dark regions.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a clean black mask.
    clean_mask = np.zeros_like(gray)

    for contour in contours:

        # Ignore tiny dark objects.
        if cv2.contourArea(contour) < MIN_OBJECT_AREA:
            continue

        # Fill in substantial dark objects.
        cv2.drawContours(
            clean_mask,
            [contour],
            -1,
            255,
            thickness=-1
        )

    return clean_mask


# STEP 3: OPEN WEBCAM

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

success, old_frame = camera.read()

if not success:
    camera.release()
    raise RuntimeError("Could not capture first frame.")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Start without any tracked points.
points = None

# Store green movement trails separately from detection masks.
trail_mask = np.zeros_like(old_frame)

frame_count = 0

print("Black object optical flow started!")
print("Q = Quit | R = Reset | C = Clear trails")


# STEP 4: PROCESS LIVE WEBCAM

while True:

    success, frame = camera.read()

    if not success:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Generate a fresh dark-object mask for every frame.
    dark_mask = create_dark_mask(frame)


    # STEP 5: TRACK EXISTING FEATURES

    if points is not None and len(points) > 0:

        # Estimate where tracked points moved.
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3
        )

        valid_new = []
        valid_old = []

        if new_points is not None and status is not None:

            for new, old, tracked in zip(
                new_points, points, status.flatten()
            ):

                # Discard features Lucas-Kanade couldn't track.
                if tracked != 1:
                    continue

                x, y = new.ravel()
                x_int, y_int = int(round(x)), int(round(y))

                # Discard points outside image boundaries.
                if not (
                    0 <= x_int < gray.shape[1]
                    and 0 <= y_int < gray.shape[0]
                ):
                    continue

                # Only keep tracked points inside dark regions.
                if dark_mask[y_int, x_int] == 0:
                    continue

                valid_new.append(new.reshape(2))
                valid_old.append(old.reshape(2))

        # Convert lists back into NumPy arrays.
        good_new = np.array(
            valid_new, dtype=np.float32
        ).reshape(-1, 2)

        good_old = np.array(
            valid_old, dtype=np.float32
        ).reshape(-1, 2)


        # STEP 6: DRAW MOVEMENT TRAILS

        for new, old in zip(good_new, good_old):

            new_x, new_y = new
            old_x, old_y = old

            a, b = int(new_x), int(new_y)
            c, d = int(old_x), int(old_y)

            # Green line shows movement between frames.
            cv2.line(
                trail_mask,
                (c, d),
                (a, b),
                (0, 255, 0),
                2
            )

        # Save successfully tracked points for the next frame.
        points = good_new.reshape(-1, 1, 2)

    else:
        points = None


    # STEP 7: PERIODICALLY DETECT NEW CORNERS

    # This allows the mouse to enter the frame at any time.
    if (
        points is None
        or len(points) < 20
        or frame_count % DETECT_INTERVAL == 0
    ):

        # Only look for new features inside dark regions.
        detection_mask = dark_mask.copy()

        # Avoid detecting multiple features at the same spot.
        if points is not None:
            for point in points:
                x, y = point.ravel()

                cv2.circle(
                    detection_mask,
                    (int(x), int(y)),
                    12,
                    0,
                    -1
                )

        # Detect additional Shi-Tomasi corners.
        new_corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=0.01,
            minDistance=7,
            mask=detection_mask
        )

        if new_corners is not None:

            if points is None or len(points) == 0:
                points = new_corners
            else:
                # Add newly found corners to existing tracks.
                points = np.concatenate(
                    (points, new_corners),
                    axis=0
                )

                # Limit total feature count.
                points = points[:MAX_CORNERS]


    # STEP 8: DRAW CURRENT FEATURE POSITIONS

    output = cv2.add(frame, trail_mask)

    if points is not None:

        for point in points:

            x, y = point.ravel()

            cv2.circle(
                output,
                (int(x), int(y)),
                4,
                (0, 255, 0),
                -1
            )

        tracked_count = len(points)

    else:
        tracked_count = 0

    # Display current tracking information.
    cv2.putText(
        output,
        f"Tracked corners: {tracked_count}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # STEP 9: DISPLAY BOTH WINDOWS

    cv2.imshow("Black Mouse - Optical Flow", output)

    # White = detected dark object
    # Black = ignored pixels
    cv2.imshow("Dark Object Mask", dark_mask)


    # STEP 10: PREPARE FOR NEXT FRAME

    old_gray = gray.copy()


    # STEP 11: KEYBOARD CONTROLS

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):

        # Completely restart feature tracking.
        points = None
        trail_mask = np.zeros_like(frame)

        print("Tracking reset!")

    elif key == ord("c"):

        # Remove trails but keep tracked points.
        trail_mask = np.zeros_like(frame)

        print("Trails cleared!")


# STEP 12: CLEAN UP

camera.release()
cv2.destroyAllWindows()
