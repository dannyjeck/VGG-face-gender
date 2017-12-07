# This script shows how to reconstruct from Caffenet features
#
# Alexey Dosovitskiy, 2015

import caffe
import numpy as np
import os
import sys
import scipy.misc
import scipy.io


# set up the inputs for the net: 
batch_size = 1
image_size = (3,224,224)
images = np.zeros((batch_size,) + image_size, dtype='float32')

# use crops of the cat image as an example 
in_image = scipy.misc.imread('ak.png')
for ni in range(images.shape[0]):
  images[ni] = np.transpose(in_image, (2,0,1))


  
# RGB to BGR, because this is what the net wants as input
data = images[:,::-1] 

# subtract the ImageNet mean
image_mean = np.reshape([93.5940, 104.7624, 129.1863],[1,3,1,1])
data -= image_mean # mean is already BGR

#initialize the caffenet to extract the features
caffe.set_mode_cpu() # replace by caffe.set_mode_gpu() to run on a GPU
vggfacenet = caffe.Net('VGG_FACE_deploy.prototxt', 'VGG_FACE.caffemodel', caffe.TEST)

# run caffenet and extract the features
vggfacenet.forward(data=data)
feat = np.copy(vggfacenet.blobs[net_name.split('_')[0]].data)


