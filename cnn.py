# Adapted from https://www.tensorflow.org/tutorials/layers

from fruit_loader import fruit_loader
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

# Uncomment to use CPU instead of GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

#######################################
#####  Hyperparameters  ##############
#######################################

start = 10
number_of_fruits = 10
learning_rate = 8e-05
keep_prob_hyperparameter = 1
batch_size = 50
num_epochs = 100

print("\n CNN HYPERPARAMETERS:")
print("batch_size:  ", batch_size)
print("num_epochs:  ", num_epochs)
print("number_of_fruits:  ", number_of_fruits)
print("learning_rate:  ", learning_rate)
print("keep_prob_hyperparameter:  ", keep_prob_hyperparameter)
print("\n")


#######################################
#####  Load in data  ##################
#######################################


print("loading data")
test_images, test_labels = fruit_loader('test', start, number_of_fruits)
train_images, train_labels = fruit_loader('train', start, number_of_fruits)

# plt.imshow(test_images[0], interpolation='nearest')
# plt.show()
# exit()


# function for reading in a batch of data
def next_batch(array, batch_size, index):
    index = index + 1
    length = len(array)
    if (index*batch_size)+batch_size >= length:
        index = (length/batch_size) % index
    batch = array[index*batch_size:(index*batch_size)+batch_size]
    return batch



#######################################
#####  Build neural net  ##############
#######################################

print("initalizing network")

x = tf.placeholder(tf.float32, shape=[None, 45, 45, 3])
labels = tf.placeholder(tf.float32, shape=[None, number_of_fruits])

conv1 = tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_flat = tf.reshape(pool2, [-1, 11 * 11 * 64])

# Dense Layer
keep_prob = tf.placeholder(tf.float32)
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.nn.dropout(x=dense, keep_prob=keep_prob)

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=number_of_fruits)
prediction = tf.argmax(logits,1)
right_answer = tf.argmax(labels,1)

# Loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.global_variables_initializer()

with tf.Session() as sess:
  init.run()

  print ("starting train")
  batch = None
  for epoch in range(num_epochs):
      for i in range(len(train_images) // batch_size):
      	batch = (next_batch(train_images, 250, i), next_batch(train_labels, 250, i))

      	train_step.run(feed_dict={x: batch[0], labels: batch[1], keep_prob: keep_prob_hyperparameter})

      ## Full info:
      # print("Epoch: %g"%epoch)
      # print("train accuracy %g"%accuracy.eval(feed_dict={x:batch[0], labels: batch[1], keep_prob: 1.0}))
      # print("test accuracy %g"%accuracy.eval(feed_dict={x: test_images, labels: test_labels, keep_prob: 1.0}))

      ## easy printing:
      print("%g\t%g"%(accuracy.eval(feed_dict={x:batch[0], labels: batch[1], keep_prob: 1.0}), accuracy.eval(feed_dict={x: test_images, labels: test_labels, keep_prob: 1.0})))

  print("DONE")
  sess.close()
