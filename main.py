
import cv2
import numpy as np
import time
from threading import Thread
from pose_detection_mediapipe.detect_pose import detect_pose
from pose_classification.pose_classification_manual_by_angle import add_classification_result

# Create Camera Object for Live Cam View
class ThreadedCamera(object):
   def __init__(self, src):
      self.capture = cv2.VideoCapture(src)
      self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

      self.FPS = 1 / 30
      self.FPS_MS = int(self.FPS * 1000)

      self.thread = Thread(target=self.update, args=())
      self.thread.daemon = True
      self.thread.start()

   def update(self):
      while True:
         if self.capture.isOpened():
               (self.status, self.frame) = self.capture.read()
         time.sleep(self.FPS)

   def end(self):
      self.capture.release()

   def show_frame(self, title):
      cv2.imshow(title, self.frame)
      if cv2.waitKey(self.FPS_MS) == 13:
         return False
      return True
    

def show_pose_detection(frame: np.ndarray) -> None:
   """
   Display the pose detection result on the given frame.

   Takes a video frame, performs pose detection on it, 
   annotates the frame with the detection results, and displays the 
   annotated frame in a window titled "Face Detection".

   Args:
      frame (numpy.ndarray): The input video frame on which pose detection 
                        is to be performed.

   Returns:
      None
   """
   #  From the given frame, show the pose detection result 
   detection_result, annotated_frame = detect_pose(frame)
   annotated_frame = add_classification_result(annotated_frame, detection_result)
   cv2.imshow("Face Detection", annotated_frame)
   return

def get_from_capture(media):
   cap1 = ThreadedCamera(media)
   while True:
      try:
         if not cap1.status:
               break
         show_pose_detection(cap1.frame)
         # Press Enter to exit
         if cv2.waitKey(cap1.FPS_MS) == 13:
               break
      except AttributeError:
         pass
   cap1.end()
   cv2.destroyAllWindows()
   return

if __name__ == "__main__":
   get_from_capture(0)