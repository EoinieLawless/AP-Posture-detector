# AP-Posture-detector

## Still W.I.P

## Shoulder-Ear Angle Monitoring System
This Python script uses the [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose) library to monitor the alignment of a person's shoulders to ears in a webcam feed. The goal is to ensure that the user maintains a straight posture for squatting.

### Basic Methods
The script defines two basic methods:

*findDistance*(x1, y1, x2, y2): Calculates the Euclidean distance between two points.
*findAngle*(x1, y1, x2, y2): Computes the angle between two points using the arccosine function.

### Warning System
A warning system is implemented to alert the user if their shoulder-ear angle exceeds a threshold, indicating poor posture.



### Visualization
The user's shoulder, ear, and hip landmarks are visualized on the webcam feed, and lines are drawn to represent the posture alignment. The current shoulder-ear angle is displayed on the video feed.

If the shoulder-ear angle surpasses 30 degrees for more than 3 seconds, a warning is displayed on the video feed.

The script uses the MediaPipe Pose library to detect and analyze key landmarks on the user's body, such as shoulders, ears, and hips.

### Execution
The script continuously captures video frames from the webcam, processes the pose landmarks, and evaluates the shoulder-ear angle to provide real-time feedback. 

### Pressing 'q' terminates the program.
