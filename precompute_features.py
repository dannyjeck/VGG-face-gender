import tensorflow as tf
import numpy as np

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

def fc7_model(x,weight_dict):
	#Input is a filename
	#Output is the fc7 output of the VGG_face model
	
	#Tensors to load and format the image
	image_string = tf.read_file(x)
	testimage = tf.image.decode_image(image_string, channels=3)
	testimage = tf.cast(testimage, tf.float32)
	averageImg = tf.constant([129.1863,104.7624,93.5940])
	averageImg = tf.reshape(averageImg, [1, 1, 1, 3])
	in_shape = [816, 816, 3]
	testimage = tf.reshape(testimage-averageImg,[1] + in_shape) #subtract the mean
	testimage = tf.image.resize_bilinear(testimage, [224, 224]) #redundant but will need later
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
	
	
	return fc7

def get_files_labels(datafile,datadir):
	#read in filenames and labels for a given fold


	with open(datafile) as f:
		lines=f.readlines()
	
	filenames = [datadir+l.split(' ')[0] for l in lines]
	labels = [int(l.split(' ')[1]) for l in lines]
	
	return (filenames,labels)


DATADIR = './Data/aligned/'
TRAIN_DATAFILE = './Data/Folds/train_val_txt_files_per_fold2/test_fold_is_0/gender_train.txt'
VAL_DATAFILE = './Data/Folds/train_val_txt_files_per_fold2/test_fold_is_0/gender_val.txt'
TEST_DATAFILE = './Data/Folds/train_val_txt_files_per_fold2/test_fold_is_0/gender_test.txt'

#Load the files
(tr_files, tr_labels) = get_files_labels(TRAIN_DATAFILE,DATADIR)
(val_files, val_labels) = get_files_labels(VAL_DATAFILE,DATADIR)
(test_files, test_labels) = get_files_labels(TEST_DATAFILE,DATADIR)


#build the model
weightfile = './Models/VGG_face_TF/VGG_face_TF.npy'

x = tf.placeholder(tf.string)
weight_dict = np.load(weightfile).item()
fc7 = fc7_model(x,weight_dict)

#Run and save features
Nfeats = 4096
with tf.Session() as sess:
		
	# Run on training data
	print('Precomputing training features')
	N_tr = len(tr_files)
	tr_out = np.zeros((N_tr,Nfeats))
	for i in range(N_tr):
		tr_out[i,:] = fc7.eval(feed_dict={x:tr_files[i]})
		if i%100==0:
			print (i, N_tr)
	np.save('./Data/tr_feats.npy',tr_out)
	np.save('./Data/tr_labels.npy',tr_labels)
	

	# Run on validation data
	print('Precomputing validation features')
	N_val = len(val_files)
	val_out = np.zeros((N_val,Nfeats))
	for i in range(N_val):
		val_out[i,:] = fc7.eval(feed_dict={x:val_files[i]})
		if i%100==0:
			print (i, N_val)
	np.save('./Data/val_feats.npy',val_out)
	np.save('./Data/val_labels.npy',val_labels)
	
	# Run on test data
	print('Precomputing test features')
	N_test = len(test_files)
	test_out = np.zeros((N_test,Nfeats))
	for i in range(N_test):
		test_out[i,:] = fc7.eval(feed_dict={x:test_files[i]})
		if i%100==0:
			print (i, N_test)
	np.save('./Data/test_feats.npy',test_out)
	np.save('./Data/test_labels.npy',test_labels)
