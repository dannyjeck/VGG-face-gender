import models
import tensorflow as tf
import numpy as np

# Get the data specifications for the GoogleNet model
spec = models.get_data_spec(model_class=models.VGG_FACE_16_layer)
	

x = tf.placeholder(tf.string) #input filename
image_string = tf.read_file(x)
testimage = tf.image.decode_image(image_string, channels=3)
testimage = tf.cast(testimage, tf.float32)
averageImg = tf.constant([129.1863,104.7624,93.5940])
averageImg = tf.reshape(averageImg, [1, 1, 3])

testimage = tf.reshape(testimage-averageImg,[-1, 224, 224, 3]) #subtract the mean
#testimage = tf.transpose(testimage, perm=[0, 1, 2, 3])
#y_ = tf.placeholder(tf.float32, shape=[None, ylen]) #labels


x_img_channel_swap = testimage[:,:,:,::-1] #RGB to BGR because the model is from caffe


# Construct the network
net = models.VGG_FACE_16_layer({'data': x_img_channel_swap})


testfile = '/home/djecklocal/Documents/python/FaceGender/Models/VGG_Face_Caffe/ak.png'
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
prediction = testimage.eval(feed_dict={x: testfile})

print(prediction)
print((np.argmax(prediction),np.max(prediction)))


