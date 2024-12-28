import math
from mediapipe import solutions
import cv2
import numpy as np

mp_pose = solutions.pose

def calculate_angle(landmark1, landmark2, landmark3) -> float:
    """
    Calculate the angle between the lines formed by three landmarks.
    Args:
        landmark1: The first landmark, with attributes x, y, and z.
        landmark2: The second landmark, with attributes x, y, and z.
        landmark3: The third landmark, with attributes x, y, and z.
    Returns:
        float: The calculated angle in degrees, constrained to be between 0 and 180 degrees.
    """
    # Calculate the angle between the lines formed by the 3 landmarks
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Ensure the angle is between 0 and 360
    if angle < 0:
        angle += 360

    # Ensure the angle is between 0 and 180 (Only interested in the acute angle)
    if angle > 180:
        angle = 360 - angle

    return angle

def classify_pose(landmarks: list) -> dict:
    """
    Classifies the pose based on the angles of the knees and waist.
    The possible poses are:
        Full Lunge: Both knees are bent, and one of the waist angles is straight.
        Half Lunge (L): Left knee is bent, right knee is straight.
        Half Lunge (R): Right knee is bent, left knee is straight.
        Unknown Pose: None of the above poses.
    Parameters:
    landmarks (list): A list of landmarks containing the coordinates of various body parts.
    Returns:
    dict: A dictionary containing the pose label and the calculated angles for the left knee, right knee, left waist, and right waist.
        - label (str): The classified pose label ('Half Lunge (L)', 'Half Lunge (R)', 'Full Lunge', or 'Unknown Pose').
        - left_knee_angle (float): The angle of the left knee.
        - right_knee_angle (float): The angle of the right knee.
        - left_waist_angle (float): The angle of the left waist.
        - right_waist_angle (float): The angle of the right waist.
    """
    # Get the landmarks
    # Left Knee, Right Knee, Left Waist, Right Waist
    label = "Unknown Pose"
    left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
    
    right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

    left_waist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], 
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP], 
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
    
    right_waist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
    
    if (left_knee_angle > 75 and left_knee_angle < 115) and (right_knee_angle > 75 and right_knee_angle < 115):
        # Both knees are bent -> can be a squat, or user is bending down
        # A full Lunge must have one of the waist angles straight
        if (left_waist_angle > 160 and left_waist_angle < 180) or (right_waist_angle > 160 and right_waist_angle < 180):
            label = "Full Lunge"
        else:
            label = "Unknown Pose"
    elif (left_knee_angle > 120 and left_knee_angle < 180) and (right_knee_angle >= 115 and right_knee_angle < 145):
        # Right knee is bent
        label = "Half Lunge (R)"
    if (left_knee_angle >= 115 and left_knee_angle < 145) and (right_knee_angle > 120 and left_knee_angle < 180):
        # Left knee is bent
        label = "Half Lunge (L)"
    return {
        "label": label,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_waist_angle": left_waist_angle,
        "right_waist_angle": right_waist_angle
    }

def add_classification_result(frame: np.ndarray, detection_result: object) -> np.ndarray:
    """
    Adds classification results to the given frame.
    This function takes a video frame and the detection results from a pose detection model,
    classifies the detected pose, and overlays the classification label and angles of specific
    joints (left knee, right knee, left waist, right waist) onto the frame.
    Args:
        frame (numpy.ndarray): The video frame to which the classification results will be added.
        detection_result (object): The result from the pose detection model, which includes pose landmarks.
    Returns:
        numpy.ndarray: The video frame with the classification results overlaid.
    """
    if len(detection_result.pose_landmarks) > 0:
        # Get the classification result if there is a pose detected
        pose_classification = classify_pose(detection_result.pose_landmarks[0])

        # Write text description on the frame
        cv2.putText(frame, pose_classification["label"], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Knee Angle: {pose_classification['left_knee_angle']:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Right Knee Angle: {pose_classification['right_knee_angle']:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Left Waist Angle: {pose_classification['left_waist_angle']:.2f}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Right Waist Angle: {pose_classification['right_waist_angle']:.2f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame