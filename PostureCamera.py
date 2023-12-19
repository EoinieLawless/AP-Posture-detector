import cv2
import time
import math as m
import mediapipe as mp
import csv
import os

# --------------------------------------------------------------------------------------------- 

# Basic methods

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

# Sends warning if posture is bad 
def sendWarning(x):
    pass

# --------------------------------------------------------------------------------------------- 

# All initializations 

# Font type for OpenCV.
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize frame counters.
good_frames = 0
bad_frames = 0

# Colors.
blue = (255, 127, 0)
purple = (128, 0, 128)
red = (50, 50, 255)
green = (127, 255, 0)
orange = (255, 165, 0)
dark_blue = (127, 20, 0)
light_blue = (173, 216, 230)
light_green = (127, 233, 100)
gray = (128, 128, 128) 
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# CSV file setup
script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
csv_filename = os.path.join(script_dir, "neck_angle_data.csv")
csv_header = ["Timestamp", "Neck Angle"]

# Write header to the CSV file
with open(csv_filename, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

# --------------------------------------------------------------------------------------------- 

if __name__ == "__main__":
    
    
    
    # For webcam input.
    cap = cv2.VideoCapture(0)
    
    qstart_time = time.time()
    
    next_write_time = 3  # Next time to write data
    start_time = time.time()

    while cap.isOpened():
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        # Get height and width of the frame for calculations.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Process the image and detect pose.
        keypoints = pose.process(image)

        # Convert back to BGR for OpenCV rendering.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark
            
            # Gathering the landmarks of the body
            # Extract landmark coordinates
            left_Shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            left_Shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Calculate distance and angles
            offset = findDistance(left_Shoulder_x, left_Shoulder_y, r_shldr_x, r_shldr_y)
            neck_inclination = findAngle(left_Shoulder_x, left_Shoulder_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, left_Shoulder_x, left_Shoulder_y)

            # --------------------------------------------------------------------------------
            
            # Drawing on the video

            # Draw landmarks.
            cv2.circle(image, (left_Shoulder_x, left_Shoulder_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (left_Shoulder_x, left_Shoulder_y - 100), 7, yellow, -1)  # For angle display
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)  # For angle display

            # Put text, Posture and angle inclination.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

            # Determine whether good posture or bad posture.
            if neck_inclination < 40 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1

                # Display posture info
                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                cv2.putText(image, str(int(neck_inclination)), (left_Shoulder_x + 10, left_Shoulder_y), font, 0.9, light_green, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

                # Join landmarks - They will turn green for good posture
                cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (l_ear_x, l_ear_y), green, 4)
                cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (left_Shoulder_x, left_Shoulder_y - 100), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (left_Shoulder_x, left_Shoulder_y), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

            else:
                good_frames = 0
                bad_frames += 1

                # Display posture info
                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                cv2.putText(image, str(int(neck_inclination)), (left_Shoulder_x + 10, left_Shoulder_y), font, 0.9, red, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

                # Join landmarks - They will turn red for bad posture
                cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (l_ear_x, l_ear_y), red, 4)
                cv2.line(image, (left_Shoulder_x, left_Shoulder_y), (left_Shoulder_x, left_Shoulder_y - 100), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (left_Shoulder_x, left_Shoulder_y), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

            # Calculate the time of remaining in a particular posture
            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            # Display posture time
            if good_time > 0:
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

            # Send a warning if bad posture time exceeds a threshold
            if bad_time > 10:
                sendWarning(bad_time)
                
            # For printing out the data collection    
            if elapsed_time >= next_write_time:
                with open(csv_filename, mode='a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([current_time, neck_inclination])
                next_write_time += 3  # Schedule the next write


            # Display the processed image
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
