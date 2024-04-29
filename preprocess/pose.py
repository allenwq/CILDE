import sys
import os

# Add the parent directory of the current script to sys.path
script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
project_dir = os.path.dirname(script_dir)  # Gets the parent directory (project/)
sys.path.append(project_dir)

import torch
import json as j
import numpy as np
from annotator.openpose import OpenposeDetector
import cv2

def main():
    body_estimation = OpenposeDetector()
    base_folder = 'data/COCO/'
    pose_folder = base_folder + 'pose/'
    origin_data_folder = base_folder + 'origin/'
    json_folder = base_folder + 'json/'

    # Check if output directories exists, if not, create them
    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    # Loop over all images in the original data folder
    for filename in os.listdir(origin_data_folder):
        if filename.endswith(".jpg"):  # Ensure we're processing .jpg files
            print('start process ' + filename)
            img_path = os.path.join(origin_data_folder, filename)
            oriImg = cv2.imread(img_path)  # B,G,R order

            # Use Openpose to detect pose 
            canvas, json_data = body_estimation(oriImg, hand=True)

            # Prepare file names (1.jpg => 1)
            name_without_ext = os.path.splitext(filename)[0] 

            # Save the pose to png
            canvas_path = os.path.join(pose_folder, f"{name_without_ext}.png")
            cv2.imwrite(canvas_path, canvas)

            # Save json data to json file
            #json_path = os.path.join(json_folder, f"{name_without_ext}.json")
            #with open(json_path, 'w') as json_file:
            #    j.dump(json_data, json_file)
    print("Processing images completed.")
main()
