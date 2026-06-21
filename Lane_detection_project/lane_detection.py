import cv2
import numpy as np

# Select only the road area
def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]

    polygons = np.array([
        [
            (0, height),
            (width, height),
            (int(width * 0.65), int(height * 0.6)),
            (int(width * 0.35), int(height * 0.6))
        ]
    ])

    mask = np.zeros_like(img)

    cv2.fillPoly(mask, polygons, 255)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


# Load video
video = cv2.VideoCapture("videos/road.mp4")

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Resize for smoother processing
    frame = cv2.resize(frame, (960, 540))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Keep only road region
    cropped_edges = region_of_interest(edges)

    # Detect lines
    lines = cv2.HoughLinesP(
        cropped_edges,
        1,
        np.pi / 180,
        50,
        minLineLength=50,
        maxLineGap=100
    )

    # Draw detected lane lines
    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            # Avoid division by zero
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Ignore nearly horizontal lines
            if abs(slope) < 0.5:
                continue

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4
            )

    cv2.imshow("Lane Detection", frame)

    # Press Q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()