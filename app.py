import os
import cv2
from cvzone import overlayPNG
from cvzone.PoseModule import PoseDetector
import mediapipe as mp

cap = cv2.VideoCapture(0)  # Use default webcam
detector = PoseDetector(detectionCon=0.5)

shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 225 / 130  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581/ 440
imageNumber = 0
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

# Resize button images to match the overlay region size
overlay_height = 75  # Adjust as needed
overlay_width = 75  # Adjust as needed
imgButtonRight = cv2.resize(imgButtonRight, (overlay_width, overlay_height))
imgButtonLeft = cv2.resize(imgButtonLeft, (overlay_width, overlay_height))

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    try:
        img = detector.findPose(img, draw=False)
        lmList, _ = detector.findPosition(img, bboxWithHands=False, draw=False)

        if lmList and len(lmList) >= 17:
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]
            imgShirtPath = os.path.join(shirtFolderPath, listShirts[imageNumber])
            imgShirt = cv2.imread(imgShirtPath, cv2.IMREAD_UNCHANGED)

            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 130
            offset = int(44 * currentScale), int(48 * currentScale)

            # Overlay the shirt image
            try:
                img = overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except Exception as e:
                print("Error overlaying shirt:", str(e))

            # Calculate the position for the right button
            button_right_x = img.shape[1] - overlay_width - 20
            button_right_y = int(img.shape[0] / 2) - overlay_height // 2

            # Calculate the position for the left button
            button_left_x = 20
            button_left_y = int(img.shape[0] / 2) - overlay_height // 2

            # Overlay the right button image
            img = overlayPNG(img, imgButtonRight, (button_right_x, button_right_y))

            # Overlay the left button image
            img = overlayPNG(img, imgButtonLeft, (button_left_x, button_left_y))

            if lmList[15][1] > 0.8 * img.shape[1]:  # Adjust threshold as needed for the right button
                counterRight += 1
                cv2.ellipse(img, (button_right_x + overlay_width // 2, button_right_y + overlay_height // 2),
                            (33, 33), 0, 0, counterRight * 10, (0, 255, 0), 10)
                if counterRight * 10 > 360:
                    counterRight = 0
                    imageNumber += 1
                    if imageNumber >= len(listShirts):
                        imageNumber = 0
            elif lmList[16][1] < 0.2 * img.shape[1]:  # Adjust threshold as needed for the left button
                counterLeft += 1
                cv2.ellipse(img, (button_left_x + overlay_width // 2, button_left_y + overlay_height // 2),
                            (33, 33), 0, 0, counterLeft * 10, (0, 255, 0), 10)
                if counterLeft * 10 > 360:
                    counterLeft = 0
                    imageNumber -= 1
                    if imageNumber < 0:
                        imageNumber = len(listShirts) - 1
            else:
                counterRight = 0
                counterLeft = 0

    except Exception as e:
        print("Error processing pose:", str(e))

    cv2.namedWindow("Virtual Trial Room", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Virtual Trial Room", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Virtual Trial Room", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()