from controlnet_aux.mesh_graphormer import MeshGraphormerDetector
import comfy.model_management as model_management

apply_midas = MeshGraphormerDetector.from_pretrained().to(model_management.get_torch_device())
from PIL import Image
import numpy as np
import os


def gen_depth_image(path_to_original):
    # load original image
    img = Image.open(path_to_original)
    detected_map = apply_midas(img)
    return detected_map[0] # shape: (Height, Width)

directory = '/mnt/d/COCO/origin'
directory_processed = '/mnt/d/COCO/hand_depth'
# Make sure the processed directory exists
os.makedirs(directory_processed, exist_ok=True)

# load all photos in the directory
images = os.listdir(directory)

for image_name in images:
    # Construct the full file path
    image_path = os.path.join(directory, image_name)
    # Check if it's a file to avoid processing directories
    if os.path.isfile(image_path):
        dpth = gen_depth_image(image_path)
        if dpth is None:
            continue
        # Convert depth map to PIL Image
        depth_image = dpth
        # Construct the save path
        save_path = os.path.join(directory_processed, image_name)
        # Save the depth image
        depth_image.save(save_path)
        print(f"Depth image saved for {image_name}")
