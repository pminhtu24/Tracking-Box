import cv2

cap = cv2.VideoCapture("highwayvn.mp4")

# Object detection from a stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)

while True:
    ret, frame = cap.read()

    # Extract region of interest
    roi = frame[150:720, 500:1000]

    # Object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and merge contours
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 90:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frames
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(2)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
