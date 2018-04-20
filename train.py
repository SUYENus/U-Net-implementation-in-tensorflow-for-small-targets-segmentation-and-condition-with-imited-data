
from tf_unet import unet, util, image_util
import cv2
import os
import numpy as np

output_path = '4.10_6L/'

#setup & training
print('setting up UNet')
batch_size = 1
net = unet.Unet(channels=3, n_class=3,layers=6, features_root=64,norm=True,cost_kwargs=dict(class_weights=[2.90, 0.36, 0.0111]))
trainer = unet.Trainer(net,learning_rate=0.1,batch_size=batch_size, optimizer='momentum', n_class=3,opt_kwargs=dict(decay_rate=0.95,momentum=0.99))
path = trainer.train(output_path, training_iters=20, epochs=1, dropout=1, restore=True, SGD=True)
print(path)
