# CILDE
Controlled Image Generation Application via LoRA-Enhanced Diffusion Models

# Training LoRA model


# Training ControlNet Models

## Setup the env
1.
```
$ conda env create --name control -file=training-env.yml
$ conda activate control
```
2. Download stable diffusion 1.5 from hugging face and save it to `models/v1-5-pruned-emaonly.ckpt`
## Training the Style Control Model
Running the necessary preprocessing scripts. (Since this dataset is small, we already provided in the data/ folder)
Download an existing softedge model from https://huggingface.co/lllyasviel/control_v11p_sd15_softedge and save it to `models/control_v11p_sd15_softedge.pth`. Our this training will be doing fine-tuning on this model
```
$ python style_train.py

```

## Train the Posture Control Model
Download an existing OpenPose model from https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_openpose.pth and save it to `models/control_sd15_openpose.pth`
Running the necessary preprocessing scripts. See next section
```
$ python pose_train.py
```

# Data Preprocessing
All data preprocessing scripts are inside `preprocess/`. The path are specified in each of the script. You may want to modify the path to your actual image folder before running the scripts.

```
# Generate the edges for the images.
$ python preprocess/gen_softedge_pidinet.py
```
```
# Generate the hand depth for the images.
$ python preprocess/gen_hand_depth.py
```

```
# Generate the posture points uing OpenPose for the images.
$ python preprocess/pose.py
```

```
# Generate the prompts for the images (OpenAI CLIP library was used)
$ python python preprocess/img-to-prompt/main.py
```


