from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import cv2

mp_pose = solutions.pose
base_options = python.BaseOptions(model_asset_path='pose_detection_mediapipe/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image: mp.Image, detection_result: object) -> np.ndarray:
  """
  Draw pose landmarks on an RGB image using MediaPipe library. The returned image will have the landmarks drawn on it.
  Args:
    rgb_image (mediapipe.Image): The input RGB image on which landmarks will be drawn.
    detection_result (object): The detection result containing pose landmarks.
  Returns:
    annotated_image (numpy.ndarray): The annotated image with pose landmarks drawn on it.
  """
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def detect_pose(frame: np.ndarray) -> tuple:
  """
  Detects poses in the given frame using MediaPipe and returns the detection results along with the frame with landmarks drawn.

  Args:
    frame (numpy.ndarray): The input image frame in BGR format.

  Returns:
    tuple: A tuple containing:
      - detection_result (PoseLandmarkerResult): The result of the pose detection.
      - annotated_image (numpy.ndarray): The input frame with detected landmarks drawn on it.
  """
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
  detection_result = detector.detect(mp_frame)
  return detection_result, draw_landmarks_on_image(frame, detection_result)