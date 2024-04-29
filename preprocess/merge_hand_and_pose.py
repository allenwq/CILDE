import sys
import os

# Add the parent directory of the current script to sys.path
script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
project_dir = os.path.dirname(script_dir)  # Gets the parent directory (project/)
sys.path.append(project_dir)

from PIL import Image
import numpy as np
directory_hand_depth = '/mnt/d/COCO/hand_depth'
directory_pose = '/mnt/d/COCO/pose'
directory_pose_depth = '/mnt/d/COCO/pose_and_hand'

os.makedirs(directory_pose_depth, exist_ok=True)

def merge_pose_and_hand(path_to_pose, path_to_hand):
    # Load the images
    pose_img = Image.open(path_to_pose)
    hand_depth_img = Image.open(path_to_hand)
    hand_depth_img = hand_depth_img.resize(pose_img.size)

    # Convert images to numpy arrays
    pose_array = np.array(pose_img)
    hand_depth_array = np.array(hand_depth_img)

    # Iterate through the hand depth map and copy non-black pixels onto the pose image
    for i in range(hand_depth_array.shape[0]):
        for j in range(hand_depth_array.shape[1]):
            if not np.all(hand_depth_array[i, j] == 0):  # Check if pixel is not black
                pose_array[i, j] = hand_depth_array[i, j]

    # Convert the resulting numpy array back to an image
    merged_img = Image.fromarray(pose_array)

    # Save the merged image
    merged_img.save(os.path.join(directory_pose_depth, os.path.basename(path_to_pose)))

pose_files = os.listdir(directory_pose)

# Iterate through each pose file
for pose_file in pose_files:
    # Check if there is a corresponding hand depth file
    hand_depth_file = os.path.join(directory_hand_depth, os.path.splitext(pose_file)[0] + '.jpg')
    if os.path.exists(hand_depth_file):
        # Merge the pose and hand depth images
        merge_pose_and_hand(os.path.join(directory_pose, pose_file), hand_depth_file)
