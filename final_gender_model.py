import tensorflow as tf
import numpy as np
import argparse

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def conv_layer(x,weight_dict,name):
	'''Inputs are x=input tensor, weight_dict and name are loaded weights and layer name
	Returns the convolutional layer
	All convolution stride values are 1'''
	
	W_conv = tf.constant(weight_dict[name]['weights'],name=name+'w')
	b_conv = tf.constant(weight_dict[name]['biases'],name=name+'b')

	h_conv = tf.nn.relu(conv2d(x,W_conv)+ b_conv)
	return h_conv

def pool_layer_2x2(x):
	'''Inputs are x=input tensor
	Returns the the pooling layer
	All pooling is max pooling with 2x2 stride'''
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def fc_layer(x,weight_dict,name):
	'''inputs are x=input tensor, weight_dict and name are loaded weights and layer name
	returns the fully connected (and relu'd) layer'''
	W_fc = tf.constant(weight_dict[name]['weights'],name=name+'w')
	b_fc = tf.constant(weight_dict[name]['biases'],name=name+'b')
	
	return	tf.nn.relu(tf.matmul(x,W_fc)+b_fc)

def final_model(x,weight_file_VGG,weight_file_final):
	'''The final tensorflow model to predict gender from and image.
	Input x is a placeholder for the filename,
	weight_file_VGG contains the parameters through fc7
	weight_file_final contains the parameters for the final layer
	A softmax layer is added so that the outputs are positively valued.
	'''

	#Load in the weights
	weight_dict_VGG = np.load(weight_file_VGG).item()
	weight_dict_final = np.load(weight_file_final).item()
	
	#Tensors to load and format the image
	image_string = tf.read_file(x)
	testimage = tf.image.decode_image(image_string, channels=3)
	testimage = tf.cast(testimage, tf.float32)
	averageImg = tf.constant([129.1863,104.7624,93.5940])
	averageImg = tf.reshape(averageImg, [1, 1, 1, 3])
	in_shape = [816, 816, 3]
	testimage = tf.reshape(testimage-averageImg,[1] + in_shape) #subtract the mean
	testimage = tf.image.resize_bilinear(testimage, [224, 224])
	x_img_channel_swap = testimage[:,:,:,::-1] #RGB to BGR because the model is from caffe

	#Construct the network
	conv1_1 = conv_layer(x_img_channel_swap,weight_dict_VGG,'conv1_1')
	conv1_2 = conv_layer(conv1_1,weight_dict_VGG,'conv1_2')
	pool1   = pool_layer_2x2(conv1_2)
	conv2_1 = conv_layer(pool1,weight_dict_VGG,'conv2_1')
	conv2_2 = conv_layer(conv2_1,weight_dict_VGG,'conv2_2')
	pool2 = pool_layer_2x2(conv2_2)
	conv3_1 = conv_layer(pool2,weight_dict_VGG,'conv3_1')
	conv3_2 = conv_layer(conv3_1,weight_dict_VGG,'conv3_2')
	conv3_3 = conv_layer(conv3_2,weight_dict_VGG,'conv3_3')
	pool3   = pool_layer_2x2(conv3_3)
	conv4_1 = conv_layer(pool3,weight_dict_VGG,'conv4_1')
	conv4_2 = conv_layer(conv4_1,weight_dict_VGG,'conv4_2')
	conv4_3 = conv_layer(conv4_2,weight_dict_VGG,'conv4_3')
	pool4   = pool_layer_2x2(conv4_3)
	conv5_1 = conv_layer(pool4,weight_dict_VGG,'conv5_1')
	conv5_2 = conv_layer(conv5_1,weight_dict_VGG,'conv5_2')
	conv5_3 = conv_layer(conv5_2,weight_dict_VGG,'conv5_3')
	pool5   = pool_layer_2x2(conv5_3)
	pool5_flat = tf.reshape(pool5, [-1, 25088]) #25088 = 512*7*7
	fc6 = fc_layer(pool5_flat,weight_dict_VGG,'fc6')
	fc7 = fc_layer(fc6,weight_dict_VGG,'fc7')

	#Final layer
	W = tf.Variable(tf.constant(weight_dict_final['W']),name='W')
	b = tf.Variable(tf.constant(weight_dict_final['b']),name='b')
	final = tf.matmul(fc7,W)+b
	
	#Softmax the output
	out = tf.nn.softmax(final)
	return out

def main():	
	#parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('img_path', help='Path to an image to test')
	args = parser.parse_args()
	if (args.img_path is None):
		print_stderr('No image given')
		exit(-1)


	#Generate model
	weight_file_VGG = './Models/VGG_face_TF/VGG_face_TF.npy'
	weight_file_final = './Models/Final/final_layer_params.npy'

	
	x = tf.placeholder(tf.string)
	out = final_model(x,weight_file_VGG,weight_file_final)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())	
		#compute the output
		print
		prediction = out.eval(feed_dict={x:args.img_path})
	
	gender = ['male','female'] #maps what the output means
	print('Predicted gender is ' + gender[np.argmax(prediction)])

if __name__ == '__main__':
    main()
