import mediapipe as mp
import cv2

video = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

raised_hand = False

retval, frame = video.read()
h, w, _ = frame.shape

with mp_pose.Pose(static_image_mode=True) as pose:
    while True:
        retval, frame = video.read()
        new_frame = frame.copy()

        result = pose.process(frame)

        result2 = hands.process(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))

        if result2.multi_hand_landmarks and raised_hand:
            for hand_landmark in result2.multi_hand_landmarks:
                mp_draw.draw_landmarks(new_frame, hand_landmark, connections=mp_hands.HAND_CONNECTIONS)

            handmarks = result2.multi_hand_landmarks

            lower_landmarks = [handmarks[0].landmark[4], handmarks[0].landmark[20]]
            ring = handmarks[0].landmark[16].y
            middle_tip = handmarks[0].landmark[12].y
            middle_mid = handmarks[0].landmark[11].y
            middle_bottom = handmarks[0].landmark[10].y
            index = handmarks[0].landmark[7].y

            thumb_touch_pinky = abs(handmarks[0].landmark[4].x - handmarks[0].landmark[18].x) < 0.01

            if all(lm.y > middle_bottom > index > middle_mid > ring > middle_tip for lm in lower_landmarks) and thumb_touch_pinky:
                new_frame = cv2.putText(new_frame, "We have a volunteer as tribute!",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2, cv2.LINE_AA,
                                        False)

        red_dot = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=1)
        mp_draw.draw_landmarks(frame, landmark_list=result.pose_landmarks, landmark_drawing_spec=red_dot)
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            if landmarks[15].y < landmarks[13].y < landmarks[11].y:
                frame = cv2.putText(frame, 'Hand is raised', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA,
                                    False)
                raised_hand = True
            else:
                raised_hand = False

            if landmarks[16].y < landmarks[14].y < landmarks[12].y:
                frame = cv2.putText(frame, 'Hand is raised', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA,
                                    False)
                raised_hand = True
            else:
                raised_hand = False

        cv2.imshow('mediapipe', frame)
        cv2.imshow('mediapipe 2', new_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
