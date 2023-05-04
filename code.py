import cv2 as cv

# Function for Face Detection
def detectAndDraw(img, cascade, nestedCascade, scale):
    faces = []
    faces2 = []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to Gray Scale
    smallImg = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    smallImg = cv.equalizeHist(smallImg)

    # Detect faces of different sizes using cascade classifier
    faces = cascade.detectMultiScale(smallImg, 1.1, 2, 0 | cv.CASCADE_SCALE_IMAGE, (30, 30))

    # Draw circles around the faces
    for r in faces:
        center = (int((r[0] + r[2] * 0.5) * scale), int((r[1] + r[3] * 0.5) * scale))
        radius = int((r[2] + r[3]) * 0.25 * scale)
        cv.circle(img, center, radius, (255, 0, 0), 3, 8, 0)

        if not nestedCascade.empty():
            smallImgROI = smallImg[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
            nestedObjects = nestedCascade.detectMultiScale(smallImgROI, 1.1, 2, 0 | cv.CASCADE_SCALE_IMAGE, (30, 30))

            # Draw circles around eyes
            for nr in nestedObjects:
                center = (int((r[0] + nr[0] + nr[2] * 0.5) * scale), int((r[1] + nr[1] + nr[3] * 0.5) * scale))
                radius = int((nr[2] + nr[3]) * 0.25 * scale)
                cv.circle(img, center, radius, (255, 0, 0), 3, 8, 0)

    # Show Processed Image with detected faces
    cv.imshow("Face Detection", img)


if __name__ == '__main__':
    # VideoCapture class for playing video for which faces to be detected
    capture = cv.VideoCapture(0)
    frameCount = 0
    # PreDefined trained XML classifiers with facial features
    cascade = cv.CascadeClassifier("../../haarcascade_frontalcatface.xml")
    nestedCascade = cv.CascadeClassifier("../../haarcascade_eye_tree_eyeglasses.xml")
    scale = 1

    if capture.isOpened():
        # Capture frames from video and detect faces
        print("Face Detection Started....")
        while True:
            frameCount += 1
            ret, frame = capture.read()
            if ret:
                detectAndDraw(frame, cascade, nestedCascade, scale)
                if cv.waitKey(1) == 27:  # Press esc to exit
                    break
            else:
                print("Error capturing frame!")
                break
    else:
        print("Could not Open Camera")
    capture.release()
    cv.destroyAllWindows()
