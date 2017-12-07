import tensorflow as tf
import numpy as np

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def conv_layer(x,weight_dict,name):
	#inputs are x=input tensor, weight_dict and name are loaded weights and layer name
	#returns the convolutional layer
	#all convolution stride values are 1
	
	W_conv = tf.Variable(weight_dict[name]['weights'],name=name+'w')
	b_conv = tf.Variable(weight_dict[name]['biases'],name=name+'b')

	h_conv = tf.nn.relu(conv2d(x,W_conv)+ b_conv)
	return h_conv

def pool_layer_2x2(x):
	#inputs are x=input tensor
	#returns the the pooling later
	#all pooling is max pooling with 2x2 stride
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def fc_layer(x,weight_dict,name):
	#inputs are x=input tensor, weight_dict and name are loaded weights and layer name
	#returns the convolutional layer
	W_fc = tf.Variable(weight_dict[name]['weights'],name=name+'w')
	b_fc = tf.Variable(weight_dict[name]['biases'],name=name+'b')
	
	return	tf.nn.relu(tf.matmul(x,W_fc)+b_fc)

weight_dict = np.load('/home/djecklocal/Documents/python/FaceGender/Models/VGG_face_TF/VGG_face_TF.npy').item()

#Get image (already 224x224) formated and into a tensorflow placeholder
testfile = '/home/djecklocal/Documents/python/FaceGender/Models/VGG_Face_Caffe/ak.png'
testlabel = 2 #third line of names.txt is Aamir Khan


xlen = 224**2
#ylen = 10
im_shape = [224, 224, 3]
in_shape = [-1] + im_shape

x = tf.placeholder(tf.string) #input filename
image_string = tf.read_file(x)
testimage = tf.image.decode_image(image_string, channels=3)
testimage = tf.cast(testimage, tf.float32)
averageImg = tf.constant([129.1863,104.7624,93.5940])
averageImg = tf.reshape(averageImg, [1, 1, 3])

testimage = tf.reshape(testimage-averageImg,in_shape) #subtract the mean
testimage = tf.image.resize_bilinear(testimage, [224, 224]) #redundant but will need later
#testimage = tf.transpose(testimage, perm=[0, 1, 2, 3])
#y_ = tf.placeholder(tf.float32, shape=[None, ylen]) #labels


x_img_channel_swap = testimage[:,:,:,::-1] #RGB to BGR because the model is from caffe

#Construct the network
conv1_1 = conv_layer(x_img_channel_swap,weight_dict,'conv1_1')
conv1_2 = conv_layer(conv1_1,weight_dict,'conv1_2')
pool1   = pool_layer_2x2(conv1_2)
conv2_1 = conv_layer(pool1,weight_dict,'conv2_1')
conv2_2 = conv_layer(conv2_1,weight_dict,'conv2_2')
pool2 = pool_layer_2x2(conv2_2)
conv3_1 = conv_layer(pool2,weight_dict,'conv3_1')
conv3_2 = conv_layer(conv3_1,weight_dict,'conv3_2')
conv3_3 = conv_layer(conv3_2,weight_dict,'conv3_3')
pool3   = pool_layer_2x2(conv3_3)
conv4_1 = conv_layer(pool3,weight_dict,'conv4_1')
conv4_2 = conv_layer(conv4_1,weight_dict,'conv4_2')
conv4_3 = conv_layer(conv4_2,weight_dict,'conv4_3')
pool4   = pool_layer_2x2(conv4_3)
conv5_1 = conv_layer(pool4,weight_dict,'conv5_1')
conv5_2 = conv_layer(conv5_1,weight_dict,'conv5_2')
conv5_3 = conv_layer(conv5_2,weight_dict,'conv5_3')
pool5   = pool_layer_2x2(conv5_3)
pool5_flat = tf.reshape(pool5, [-1, 25088]) #25088 = 512*7*7
fc6 = fc_layer(pool5_flat,weight_dict,'fc6')
fc7 = fc_layer(fc6,weight_dict,'fc7')
fc8 = fc_layer(fc7,weight_dict,'fc8')
y = tf.nn.softmax(fc8, name='out')

#Run 
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
prediction = y.eval(feed_dict={x:testfile})
print(prediction)
print((np.argmax(prediction),np.max(prediction)))
if np.argmax(prediction)==testlabel:
	print("Model Copied Successfully")
