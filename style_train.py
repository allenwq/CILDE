from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch

torch.set_float32_matmul_precision('medium')


# Configs
resume_path = './models/control_v11p_sd15_softedge.pth'
origin_path = './models/v1-5-pruned-emaonly.ckpt'
#resume_path = './models/control_sd15_hed.pth'
batch_size = 3
logger_freq = 200
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/11_softedge.yaml').cpu()
#model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(origin_path, location='cpu'), strict=False)
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=23, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(devices=1, precision=32, callbacks=[logger], max_epochs=100, max_steps=3000)


# Train!
trainer.fit(model, dataloader)
