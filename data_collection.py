import math

import cv2
import mediapipe as mp
import numpy as np
import string

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'L', 'M', 'W', 'X', 'Z', '0', '1', '2', '3']
count = 1
imgSize = 500

current = classes[10]

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        try:
            success, image = cap.read()
            # image = cv2.flip(image, 1)
            h, w, c = image.shape  # for rgb c = 3

            if not success:
                print('Ignoring empty feed')
                continue

            image.flags.writeable = True
            cv2.imshow("bgr", image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
            results = hands.process(image)  # process rgb image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # put back into bgr

            annotated_image = image.copy()  # deep copy image array

            if results.multi_hand_landmarks:  #  check if hands are found
                for handType, handLandmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                    #  handType either left or right
                    #  21 landmarks in handLandmarks
                    x_list = []  # position of x
                    y_list = []  # position of y
                    lms = []
                    for landmark in handLandmarks.landmark:  # iterate over 21 landmarks
                        px, py, pz = int(landmark.x * w), int(landmark.y * h), int(landmark.z * w)  # denormalize and
                        # skip z
                        x_list.append(px)
                        y_list.append(py)

                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)

                    boxW, boxH = x_max - x_min, y_max - y_min
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    croppedImg = annotated_image[y_min - 50:y_max + 50, x_min - 50:x_max + 50]
                    h, w, _ = croppedImg.shape
                    if h > w:
                        wCal = math.ceil(imgSize * w / h)
                        croppedImg = cv2.resize(croppedImg, (wCal, imgSize))
                        pushWidth = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, pushWidth:pushWidth + wCal] = croppedImg
                    else:
                        hCal = math.ceil(imgSize * h / w)
                        croppedImg = cv2.resize(croppedImg, (imgSize, hCal))
                        pushHeight = math.ceil((imgSize - hCal) / 2)
                        imgWhite[pushHeight:pushHeight + hCal, :] = croppedImg

                    cv2.imshow("cropped", imgWhite)
                    # print(croppedImg.shape)
                    # print(classes[prediction])
                    if cv2.waitKey(1) == ord('s'):
                        cv2.imwrite(f"DATA_NEW/{current}/{count}.jpg", imgWhite)
                        count += 1

                    # mp_draw.draw_landmarks(annotated_image, handLandmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(annotated_image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f'{current} {count}', (x_min - 30, y_min - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)

                cv2.imshow("annotated", annotated_image)

            if cv2.waitKey(1) == ord('q'):
                break
        except Exception:
            pass

cap.release()
