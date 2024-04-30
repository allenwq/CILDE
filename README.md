# CILDE
Controlled Image Generation Application via LoRA-Enhanced Diffusion Models

# Training LoRA model


# Training ControlNet Models

## Setup the env
1.
```
$ conda env create --name control -file=training-env.yml
$ conda activate control
$ pip install -r requirements.txt
```
2. Download stable diffusion 1.5 from hugging face and save it to `models/v1-5-pruned-emaonly.ckpt`

   

## Training the lora

1、Go to the train/train_lora folder in the root directory. Overwrite the control.toml file in this directory as follow.

```
[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # Training resolution ,here is 512x512
batch_size = 4                              # size of batch

  [[datasets.subsets]]
  image_dir = './data/lora'                     # train image's floder,here is data/lora folder in the root 
  class_tokens = 'LiNusSg'                # Specify an identifier class
  num_repeats = 100                          # The number of trainings per training image
```

2、Go to the data/lora folder in the root directory. Prepare an image that is consistent with resolution as the training material. You can use the caption suffix file with the same name as the image as the explanatory prompt for the training.



3、Go to the root directory, then run

```
$ accelerate launch --num_cpu_threads_per_process 1 train_network.py \
    --pretrained_model_name_or_path='./models/v1-5-pruned-emaonly.ckpt' \
    --dataset_config='./train/train_lora/control.toml' \
    --output_dir='./train' \
    --output_name='lionNUS.safetensors' \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=9000 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --cache_latents \
    --gradient_checkpointing \
    --save_every_n_epochs=1 \
    --network_module=networks.lora

```

pretrained_model_name_or_path Specifies the location of the diffusion large model. dataset_config specifies the location of the previous control.toml file. output_name Specifies the output name of the model. The remaining Settings are default.





## Training the Style Control Model

Running the necessary preprocessing scripts. (Since this dataset is small, we already provided in the data/ folder)
Download an existing softedge model from https://huggingface.co/lllyasviel/control_v11p_sd15_softedge and save it to `models/control_v11p_sd15_softedge.pth`. Our this training will be doing fine-tuning on this model
```
$ python style_train.py

```

Pretrained models are avaliable for downloading here: https://huggingface.co/real-donald-trump/5242-models

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

# 3rd Party UI
You may use https://github.com/AUTOMATIC1111/stable-diffusion-webui to use our models for inference. 
Put the both ControlNet models under `models/ControlNet` and LoRA models under `models/Lora` then start the server. 
