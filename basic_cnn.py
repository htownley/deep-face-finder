# Adapted from https://www.tensorflow.org/tutorials/layers

from fruit_loader import fruit_loader
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

# Uncomment to use CPU instead of GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


print("loading data")
test_images, test_labels = fruit_loader('test')
train_images, train_labels = fruit_loader('train')



# plt.imshow(test_images[0], interpolation='nearest')
# plt.show()
# exit()

#######################################
#####  Load in data  ##################
#######################################

#######################################
#####  Build neural net  ##############
#######################################


def next_batch(array, batch_size, index):
    index = index + 1
    length = len(array)
    if (index*batch_size)+batch_size >= length:
        index = (length/batch_size) % index
    batch = array[index*batch_size:(index*batch_size)+batch_size]
    return batch


print("initalizing network")

number_of_fruits = 10

# x = tf.placeholder(tf.float32, shape=[None, 45, 45, 3])
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
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
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# Dense Layer
keep_prob = tf.placeholder(tf.float32)
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=(1-keep_prob))

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=number_of_fruits)

# Loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 50
num_epochs = 100
init = tf.global_variables_initializer()


with tf.Session() as sess:
  init.run()

  print ("starting train")
  for epoch in range(num_epochs):
      for i in range(len(train_images) // batch_size):
      	batch = (next_batch(train_images, 50, i), next_batch(train_labels, 50, i))

      	if i%100 == 0:
      		train_accuracy = accuracy.eval(feed_dict={x:batch[0], labels: batch[1], keep_prob: 0.95})
      		print("step %d, training accuracy %g"%(i, train_accuracy))
      	train_step.run(feed_dict={x: batch[0], labels: batch[1], keep_prob: 0.5})

      print("Epoch: %g"%epoch)
      print("test accuracy %g"%accuracy.eval(feed_dict={x: test_images, labels: test_labels, keep_prob: 1.0}))

  sess.close()
