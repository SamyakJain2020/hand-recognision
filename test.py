import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import mediapipe as mp

#import dlib

link = '/media/samyak/DATA/DATA/2 DO/SAMYAK/HandMotion/Dataset/Address/C0597.MP4'

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def load_video(l):
    cap = cv2.VideoCapture(l)
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            cv2.imshow('ISL', image)
            if cv2.waitKey(4) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def display_video_data(l):
    cap = cv2.VideoCapture(l)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f'Total Frames: {total_frames}, Height: {height}, Width: {width}, Frame Rate: {frame_rate}')


def mediapipe_detection(img1, model):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    img1.flags.writeable = False  # Image is no longer writeable
    results = model.process(img1)  # Make prediction
    img1.flags.writeable = True  # Image is now writeable
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return img1, results


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )


def detect_landmarks(l):
    cap = cv2.VideoCapture(l)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.9) as holistic:
        rh_landmarks = []
        lf_landmarks = []
        pose_landmarks = []

        for i in range(n_frames):
            if i % 1 == 0:
                ret, image = cap.read()
                images, results = mediapipe_detection(image, holistic)
                rh_landmarks.append(results.right_hand_landmarks)
                lf_landmarks.append(results.left_hand_landmarks)
                pose_landmarks.append(results.pose_landmarks)
                draw_landmarks(images, results)

                if ret:
                    cv2.imshow('ISL', images)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
    return rh_landmarks, lf_landmarks, pose_landmarks


if __name__ == '__main__':
    load_video(link)
    d = []
    e = []
    f = []
    d, e, f = detect_landmarks(link)
    print(type(d), len(e), len(f))
    print(len(d))
    display_video_data(link)
