import os
import shutil
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Separar em pastas
def organize(path):
  for file in os.listdir(path):
    if '.' not in file:
      continue
    
    # Construct full paths
    destination = os.path.join(path, file.split('_')[1][:-5])
    if not os.path.exists(destination):
      os.makedirs(destination)
    
    source_path = os.path.join(path, file)
    destination_path = os.path.join(destination, file)
    
    # Move the file
    shutil.move(source_path, destination_path)
    
def transformImage(path, name, func):
  temp = os.path.basename(os.path.normpath(path))
  new_path = os.path.join(os.path.dirname(path), f'{temp}-{name}')
  if not os.path.exists(new_path):
    os.mkdir(new_path)
  
  for folder in os.listdir(path):
    curr_path = os.path.join(path, folder)
    curr_new_path = os.path.join(new_path, folder)
    if not os.path.exists(curr_new_path):
      os.mkdir(curr_new_path)
    for file in os.listdir(curr_path):
      if '.' not in file:
        continue
      source_file = os.path.join(curr_path, file)
      destination_file = os.path.join(curr_new_path, file)
      img = cv.imread(input)
      img = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
      cv.imwrite(destination_file, func(img))

def binarize(input):
  _, img = cv.threshold(
    input, 127, 255, cv.THRESH_BINARY
  )
  return img
 
def fourier(input):
  return mag(fft(input))


# Transformada de Fourier
def fft(img):
  img = np.fft.fft2(img)
  img = np.fft.fftshift(img)
  return img
# Obtém a magnitude da imagem
def mag(img):
  absvalue = np.abs(img)
  magnitude = 20 * np.log(absvalue)
  return magnitude

# Inversa (retorna para imagem original)
def ifft(fimg):
  fimg = np.fft.ifftshift(fimg)
  fimg = np.fft.ifft2(fimg)
  return fimg


# Normaliza a imagem entre 0 e 255
def norm(img):
  img = cv.normalize(
    img, None, 0, 255,
    cv.NORM_MINMAX
  )

# Melhor para ver imagens da transformada e imagens pequenas em geral.
def show(img):
  plt.imshow(img, cmap='gray')
  plt.show()
  return img