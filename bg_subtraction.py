import cv2
import numpy as np
import matplotlib.pyplot as plt

# load a video
video = cv2.VideoCapture(r"data\video.mp4")
# video = cv2.VideoCapture(r'lib\object_detection\media\videos\vtest.avi')
# video = cv2.VideoCapture(0) # Webcam

# You can set custom kernel size if you want.
kernel = None

# Initialize the background object.
backgroundObject = cv2.createBackgroundSubtractorKNN(detectShadows=False)

frame_num = 0

while True:
    frame_num += 1
    # Read a new frame.
    ret, frame = video.read()

    # Check if frame is not read correctly.
    if not ret:

        # Break the loop.

        break
    c = 2
    if c % 2 == 0:
        # Apply the background object on the frame to get the segmented mask.
        fgmask = backgroundObject.apply(frame)
        # initialMask = fgmask.copy()

        # Perform thresholding to get rid of the shadows.
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        # noisymask = fgmask.copy()

        # Apply some morphological operations to make sure you have a good mask
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        # Detect contours in the frame.
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a copy of the frame to draw bounding boxes around the detected cars.
        frameCopy = frame.copy()
        idx = 0
        # loop over each contour found in the frame.
        for cnt in contours:
            idx += 1

            # Make sure the contour area is somewhat higher than some threshold to make sure its an oject and not some noise.
            if cv2.contourArea(cnt) > 400:

                # Retrieve the bounding box coordinates from the contour.
                x, y, width, height = cv2.boundingRect(cnt)

                # Export frame
                # roi = frame[y: y + height, x: x + width]
                # cv2.imwrite('lib\object_detection\media\images\output/' + str(frame_num) + '_' + str(idx) + '.jpg', roi)

                # Draw a bounding box around the car.
                cv2.rectangle(
                    frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2
                )

                # Write Car Detected near the bounding box drawn.
                cv2.putText(
                    frameCopy,
                    "Object Detected",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Extract the foreground from the frame using the segmented mask.
        foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)

        # Stack the original frame, extracted foreground, and annotated frame.
        stacked = np.hstack((frame, foregroundPart, frameCopy))

        # Display the stacked image with an appropriate title.
        cv2.imshow(
            "Original Frame, Extracted Foreground and Detected Objects",
            cv2.resize(stacked, None, fx=0.5, fy=0.5),
        )

        c += 1
    k = cv2.waitKey(30) & 0xFF
    if k == 27 or k == "q":
        break

video.release()
cv2.destroyAllWindows()
