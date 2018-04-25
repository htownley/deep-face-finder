# Adapted from https://www.tensorflow.org/tutorials/layers

# deep neural net with 256 and 64 hidden neurons
# test accuracy of 86% but inconsistent

from fruit_loader import fruit_loader
import tensorflow as tf
import numpy as np
import os

# Uncomment to use CPU instead of GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


print("loading data")
test_images, test_labels = fruit_loader('test')
train_images, train_labels = fruit_loader('train')


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

# hyperparameters
number_of_fruits = 10
learning_rate = 1e-4

print()
print("hyperparameters")
print("number_of_fruits:  ", number_of_fruits)
print("learning_rate:  ", learning_rate)
print()

x = tf.placeholder(tf.float32, shape=[None, 45, 45, 3])
labels = tf.placeholder(tf.float32, shape=[None, number_of_fruits])


x_flat = tf.reshape(x, [-1, 45 * 45 * 3])

# Dense Layer
keep_prob = tf.placeholder(tf.float32)
dense1 = tf.layers.dense(inputs=x_flat, units=256, activation=tf.nn.relu)
dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
# dense3 = tf.layers.dense(inputs=dense2, units=256, activation=tf.nn.relu)

# dropout = tf.layers.dropout(inputs=dense3, rate=(1-keep_prob))

# Logits Layer
logits = tf.layers.dense(inputs=dense2, units=number_of_fruits)

# Loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

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
      	train_step.run(feed_dict={x: batch[0], labels: batch[1], keep_prob: 0.5})

      print("Epoch: %g"%epoch)
      print("train accuracy %g"%accuracy.eval(feed_dict={x:batch[0], labels: batch[1], keep_prob: 0.95}))
      print("test accuracy %g"%accuracy.eval(feed_dict={x: test_images, labels: test_labels, keep_prob: 1.0}))

  sess.close()
