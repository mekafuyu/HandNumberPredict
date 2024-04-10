import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import model_utils

epochs = 100
batch_size = 32
patience = 16
learning_rate = 1e-3
model_path = 'checkpoints/'
train_path = 'fingers/train'

# organize(train_path)
# transformImage(train_path, 'binarize', binarize)
# transformImage(train_path, 'fourier', fourier)
# transformImage(train_path + '-binarize', 'fourier', fourier)

model1 = model_utils.DefaultModel(
  model_path + 'model1.keras',
  train_path + '-binarize-fourier',
  epochs,
  batch_size,
  patience,
  learning_rate
)
model1.CreateLoadModel()
model1.Split()
model1.Fit()