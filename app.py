import streamlit as st
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import os
from PIL import Image

def main():
    st.title("Virtual T-Shirt Try-On")
    
    # File uploader for video input
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    
    if uploaded_file is not None:
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.video(temp_file_path)
        
        cap = cv2.VideoCapture(temp_file_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1250)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        st.write(f"The width of the frame is {width}")
        st.write(f"The height of the frame is {height}")

        detector = PoseDetector()

        shirtsFolderPath = "Resources/Shirts"
        listShirts = os.listdir(shirtsFolderPath)
        fixedRatio = 262 / 190  # widthOfShirt/WidthOfPoint111012
        shirtRatioHeightWidth = 581 / 440
        ImageNumber = 0
        imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
        imgButtonLeft = cv2.flip(imgButtonRight, 1)
        counterRight = 0
        counterLeft = 0
        selectionSpeed = 10
        
        result_video_path = "result_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (int(width), int(height)))

        while True:
            success, img = cap.read()
            if not success:
                break

            img = detector.findPose(img)
            lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
            if lmList:
                lm11 = lmList[11][0:2]
                lm12 = lmList[12][0:2]
                imgShirt = cv2.imread(os.path.join(shirtsFolderPath, listShirts[ImageNumber]), cv2.IMREAD_UNCHANGED)
                imgShirt = cv2.resize(imgShirt, (0, 0), None, 0.5, 0.5)

                widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
                imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
                currentScale = (lm11[0] - lm12[0]) / 190
                offset = int(44 * currentScale), int(48 * currentScale)

                try:
                    img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
                except:
                    pass

                img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
                img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

                if lmList[16][1] < 400:
                    counterRight += 1
                    cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)

                    if counterRight * selectionSpeed > 360:
                        counterRight = 0
                        if ImageNumber < len(listShirts) - 1:
                            ImageNumber += 1

                elif lmList[15][1] > 900:
                    counterLeft += 1
                    cv2.ellipse(img, (1100, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)

                    if counterLeft * selectionSpeed > 360:
                        counterLeft = 0
                        if ImageNumber > 0:
                            ImageNumber -= 1

                else:
                    counterRight = 0
                    counterLeft = 0

            out.write(img)
        
        cap.release()
        out.release()
        st.video(result_video_path)

if __name__ == "__main__":
    main()
