import tensorflow as tf
import numpy as np

val_feats = np.load('./Data/val_feats.npy')
val_labels = np.load('./Data/val_labels.npy')

test_feats = np.load('./Data/test_feats.npy')
test_labels = np.load('./Data/test_labels.npy')

weight_dict = np.load('./Models/Final/final_layer_params.npy').item()

#Intput placeholder
x = tf.placeholder(tf.float32,shape=[None, 4096])

#dropout
keep_prob = tf.placeholder(tf.float32)
x_drop = tf.nn.dropout(x, keep_prob)

#Final layer
shape = [4096,2] #male or female output
W = tf.Variable(tf.constant(weight_dict['W']),name='W')
b = tf.Variable(tf.constant(weight_dict['b']),name='b')
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
W.assign(weight_dict['W'])
b.assign(weight_dict['b'])
val_accuracy = accuracy.eval(feed_dict={x:val_feats, y_label:val_labels, keep_prob:1.0})
print('validation accuracy %g' % (val_accuracy) )

test_accuracy = accuracy.eval(feed_dict={x:test_feats, y_label:test_labels, keep_prob:1.0})
print('test accuracy %g.' % (test_accuracy) )

print('No changes made after this was measured')

male_test_feats = test_feats[test_labels==0]
male_labels = np.zeros(male_test_feats.shape[0])
male_accuracy = accuracy.eval(feed_dict={x:male_test_feats, y_label:male_labels, keep_prob:1.0})
print('test accuracy on male images %g.' % (male_accuracy) )

female_test_feats = test_feats[test_labels==1]
female_labels = np.ones(female_test_feats.shape[0])
female_accuracy = accuracy.eval(feed_dict={x:female_test_feats, y_label:female_labels, keep_prob:1.0})
print('test accuracy on female images %g.' % (female_accuracy) )
