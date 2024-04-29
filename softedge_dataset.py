import json
import cv2
import numpy as np
import csv

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, csv_file='data/styles/prompts.csv', source_dir='data/styles/softedge', target_dir='data/styles/origin'):
        self.data = []
        self.source_dir = source_dir
        self.target_dir = target_dir
        with open(csv_file, 'rt') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                filename, prompt = row
                self.data.append({'filename': filename, 'prompt': prompt.strip()})

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = f"{self.source_dir}/{item['filename']}"
        target_filename = f"{self.target_dir}/{item['filename']}"
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Convert images from BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        w = 512
        h = 512
        # Resize and pad images
        source = self.resize_and_pad(source, w)
        target = self.resize_and_pad(target, w)
        assert source.shape[:2] == (w, h), f"Source image {item['filename']} is not 256x256 after resize and pad."
        assert target.shape[:2] == (w, h), f"Target image {item['filename']} is not 256x256 after resize and pad."



        # Normalize the source image to [0, 1]
        source = source.astype(np.float32) / 255.0

        # Normalize the target image to [-1, 1]
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    def resize_and_pad(self, img, size, pad_color=0):
        h, w = img.shape[:2]
        sh, sw = size, size
    
        # interpolation method
        if h > sh or w > sw:  # shrinking image
            interp = cv2.INTER_AREA
        else:  # stretching image
            interp = cv2.INTER_CUBIC
    
        # aspect ratio of image
        aspect = w/h
    
        # computing scaling and pad sizing
        if aspect > 1:  # horizontal image
            new_w = sw
            new_h = np.round(new_w / aspect).astype(int)
            pad_vert = (sh - new_h) // 2
            pad_top, pad_bot = pad_vert, pad_vert
            pad_left, pad_right = 0, 0
        elif aspect < 1:  # vertical image
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = (sw - new_w) // 2
            pad_left, pad_right = pad_horz, pad_horz
            pad_top, pad_bot = 0, 0
        else:  # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    
        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    
        # Adjust if the image is still not the desired size due to rounding issues
        final_h, final_w = scaled_img.shape[:2]
        if final_h != sh or final_w != sw:
            padding_to_add_h = sh - final_h
            padding_to_add_w = sw - final_w
            scaled_img = cv2.copyMakeBorder(scaled_img, 0, padding_to_add_h, 0, padding_to_add_w, cv2.BORDER_CONSTANT, value=pad_color)
    
        return scaled_img
