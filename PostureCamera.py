import cv2
import mediapipe as mp
import math as m
import time
import csv 

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
    print(f"Warning: Straighten your shoulders! Current angle: {x}")

# --------------------------------------------------------------------------------

# All initializations
font = cv2.FONT_HERSHEY_SIMPLEX
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Timer variables
bad_angle_start_time = 0
bad_angle_duration_threshold = 3  # this is in seconds

# CSV file setup
csv_filename = "shoulder_ear_angle_measurements.csv"
csv_header = ["Timestamp", "Shoulder-Ear Angle"]
csv_data = []

# --------------------------------------------------------------------------------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    #start a timer
    start_time = time.time()

    while cap.isOpened():
        
        current_time = time.time()
        elapsed_time = current_time - start_time

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
            r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * frame.shape[1])
            r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * frame.shape[0])
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * frame.shape[1])
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * frame.shape[0])

            offset = findDistance(left_Shoulder_x, left_Shoulder_y, r_shldr_x, r_shldr_y)

            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (frame.shape[1] - 150, 30), font, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (frame.shape[1] - 150, 30), font, 0.9, (0, 0, 255), 2)


            # Calculate the angle between the shoulder and the midpoint of the ears
            shoulder_ear_midpoint_x = (l_ear_x + r_ear_x) // 2
            shoulder_ear_midpoint_y = (l_ear_y + r_ear_y) // 2
            shoulder_ear_angle = findAngle(left_Shoulder_x, left_Shoulder_y, shoulder_ear_midpoint_x, shoulder_ear_midpoint_y)


            #Shoulder Ear alignment angle, to straighten the head and shoulders
            if shoulder_ear_angle > 30:
                if bad_angle_start_time == 0:
                    bad_angle_start_time = current_time
                else:
                    bad_angle_duration = current_time - bad_angle_start_time

                    if bad_angle_duration >= bad_angle_duration_threshold:
                        sendWarning(shoulder_ear_angle)
                        cv2.putText(image, 'Straighten Shoulders!', (frame.shape[1] - 200, 100), font, 0.9, (0, 0, 255), 2)
                        # Append data to the CSV list
                        csv_data.append([current_time, shoulder_ear_angle])
            else:
                bad_angle_start_time = 0
                
                
            # Draw landmarks on the camera view
            cv2.circle(image, (left_Shoulder_x, left_Shoulder_y), 7, (0, 255, 255), -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (255, 0, 255), -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, (0, 255, 255), -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)

            # Join landmarks
            cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (l_ear_x, l_ear_y), (255, 255, 0), 4)
            cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (shoulder_ear_midpoint_x, shoulder_ear_midpoint_y), (255, 255, 0), 4)
            cv2.line(image, (l_hip_x, l_hip_y), (left_Shoulder_x, left_Shoulder_y), (255, 255, 0), 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (255, 255, 0), 4)

            cv2.putText(image, str(int(shoulder_ear_angle)) + ' Shoulder-Ear Angle', (frame.shape[1] - 200, 70), font, 0.9, (255, 255, 0), 2)

            # For the Csv File 
            if elapsed_time >= 5:
                # Save data to CSV file every 5 seconds
                with open(csv_filename, mode='w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(csv_header)
                    csv_writer.writerows(csv_data)

                # Reset the timer and data list
                start_time = time.time()
                csv_data = []
            
            
            cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
