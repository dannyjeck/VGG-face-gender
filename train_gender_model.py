import tensorflow as tf
import numpy as np
import random


tr_feats = np.load('./Data/tr_feats.npy')
tr_labels = np.load('./Data/tr_labels.npy')
N_tr = len(tr_labels)

val_feats = np.load('./Data/val_feats.npy')
val_labels = np.load('./Data/val_labels.npy')

#Intput placeholder
x = tf.placeholder(tf.float32,shape=[None, 4096])

#dropout
keep_prob = tf.placeholder(tf.float32)
x_drop = tf.nn.dropout(x, keep_prob)

#Final layer
shape = [4096,2] #male or female output
W = tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W')
b = tf.Variable(tf.constant(0.1, shape=[1, 2]),name='b')
y = tf.matmul(x_drop,W)+b

#Correct labels
y_label = tf.placeholder(tf.int32,shape=[None])
y_ = tf.one_hot(y_label,2)

#Loss, Accuracy, and Training
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Run training

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) #puts all tf.Variables into memory
val_maxacc = 0.964
for k in range(20):
	for i in range(2000):
		batchsize = 64
		ind = np.random.randint(0,N_tr,size=(batchsize)) #a bit of a hack for batching the data
		tr_feats_batch = tr_feats[ind,:]
		tr_labels_batch = tr_labels[ind]
		if i % 100 ==0: #print training accuracy every 100 steps (no dropout)
			train_accuracy = accuracy.eval(feed_dict={x:tr_feats_batch, 
				y_label:tr_labels_batch, keep_prob:1.0})
			print('Epoch %d, step %d, training accuracy %g' % (k, i, train_accuracy) )
		train_step.run(feed_dict={x:tr_feats_batch, y_label:tr_labels_batch, keep_prob:0.5}) #train with dropout on
	
	#Compute validation accuracy
	val_accuracy = accuracy.eval(feed_dict={x:val_feats, y_label:val_labels, keep_prob:1.0})
	print('Epoch %d, validation accuracy %g' % (k, val_accuracy) )
	if val_accuracy > val_maxacc: #use early stopping to get the best weights out
		np.save('final_layer_params',[{'W':W.eval(), 'b':b.eval()}])
		val_maxacc = val_accuracy


