import cv2
import mediapipe as mp
import math as m

# --------------------------------------------------------------------------------
        
# Basic methods
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def sendWarning(x):
    pass

# --------------------------------------------------------------------------------
              
# All initializations 
font = cv2.FONT_HERSHEY_SIMPLEX
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# --------------------------------------------------------------------------------
        

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Null.Frames")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            left_Shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * frame.shape[1])
            left_Shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * frame.shape[0])
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * frame.shape[1])
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * frame.shape[0])
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * frame.shape[1])
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * frame.shape[0])
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * frame.shape[1])
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * frame.shape[0])

            offset = findDistance(left_Shoulder_x, left_Shoulder_y, r_shldr_x, r_shldr_y)

            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (frame.shape[1] - 150, 30), font, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (frame.shape[1] - 150, 30), font, 0.9, (0, 0, 255), 2)

            neck_inclination = findAngle(left_Shoulder_x, left_Shoulder_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, left_Shoulder_x, left_Shoulder_y)

            # Draw landmarks on the camera view
            cv2.circle(image, (left_Shoulder_x, left_Shoulder_y), 7, (0, 255, 255), -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (255, 0, 255), -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, (0, 255, 255), -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)


            # Join landmarks 
            cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (l_ear_x, l_ear_y), 4)
            cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (left_Shoulder_x, left_Shoulder_y - 100), 4)
            cv2.line(image, (l_hip_x, l_hip_y), (left_Shoulder_x, left_Shoulder_y), 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), 4)

            cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
