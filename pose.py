from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
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

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_input_point(image):
  base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_lite.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  image = mp.Image.create_from_file(image)

  detection_result = detector.detect(image)
  pose_landmarks = detection_result.pose_landmarks
  
  try:
    nose = pose_landmarks[0][0]
    right_shoulder = pose_landmarks[0][12]
    right_elbow = pose_landmarks[0][14]
    right_wrist = pose_landmarks[0][16]
    left_shoulder = pose_landmarks[0][11]
    left_elbow = pose_landmarks[0][13]
    left_wrist = pose_landmarks[0][15]
    left_hip = pose_landmarks[0][23]
    left_knee = pose_landmarks[0][25]
    left_ankle = pose_landmarks[0][27]
    right_hip = pose_landmarks[0][24]
    right_knee = pose_landmarks[0][26]
    right_ankle = pose_landmarks[0][28]
    
    right_hand = calculate_distance(right_shoulder.x*image.width, right_shoulder.y*image.height, right_elbow.x*image.width, right_elbow.y*image.height) + calculate_distance(right_elbow.x*image.width, right_elbow.y*image.height, right_wrist.x*image.width, right_wrist.y*image.height)

    left_hand = calculate_distance(left_shoulder.x*image.width, left_shoulder.y*image.height, left_elbow.x*image.width, left_elbow.y*image.height) + calculate_distance(left_elbow.x*image.width, left_elbow.y*image.height, left_wrist.x*image.width, left_wrist.y*image.height)

    right_foot = calculate_distance(right_hip.x*image.width, right_hip.y*image.height, right_knee.x*image.width, right_knee.y*image.height) + calculate_distance(right_knee.x*image.width, right_knee.y*image.height, right_ankle.x*image.width, right_ankle.y*image.height)

    left_foot = calculate_distance(left_hip.x*image.width, left_hip.y*image.height, left_knee.x*image.width, left_knee.y*image.height) + calculate_distance(left_knee.x*image.width, left_knee.y*image.height, left_ankle.x*image.width, left_ankle.y*image.height)

    coords = np.array([[nose.x*image.width,nose.y*image.height], [left_hip.x*image.width,left_hip.y*image.height]])

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    # cv2.imshow('Visualized Mask',visualized_mask)
    # Display the annotated image in RGB format
  except:
    return "Tidak ada tubuh yang terdeteksi"
  # plt.figure(figsize=(10, 10))
  # plt.imshow(annotated_image)
  # plt.title(f"Mediapipe Result")
  # plt.axis('on')
  # plt.show()  
  return coords, annotated_image, left_hand, right_hand, left_foot, right_foot

#print(get_input_point('images/avatar2.jpg'))
# get_input_point('images/baby5-up.jpeg')
