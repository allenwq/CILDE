from controlnet_aux.pidi import PidiNetDetector
import comfy.model_management as model_management

apply = PidiNetDetector.from_pretrained().to(model_management.get_torch_device())
from PIL import Image
import numpy as np
import os


def gen_depth_imagei1(path_to_original):
    # load original image
    img = Image.open(path_to_original)
    output_img = apply(img)
    return output_img

def gen_depth_image(path_to_original):
    # load original image
    img = Image.open(path_to_original)
    original_size = img.size  # Store the original size

    output_img = apply(img)

    # Resize output_img to match the original image's size
    resized_output_img = output_img.resize(original_size)

    return resized_output_img

directory = '/mnt/d/self/origin'
directory_processed = '/mnt/d/self/softedge'
# Make sure the processed directory exists
os.makedirs(directory_processed, exist_ok=True)

# load all photos in the directory
images = os.listdir(directory)
#images = sorted(images, reverse=True)

for image_name in images:
    # Construct the full file path
    image_path = os.path.join(directory, image_name)
    # Check if it's a file to avoid processing directories
    if os.path.isfile(image_path):
        save_path = os.path.join(directory_processed, image_name)
        if os.path.exists(save_path):
            print(f"File already exists at {save_path}. Skipping...")
            continue
        dpth = gen_depth_image(image_path)
        if dpth is None:
            continue
        # Convert depth map to PIL Image
        depth_image = dpth
        # Construct the save path
        depth_image.save(save_path)
        print(f"Image edge saved for {image_name}")
