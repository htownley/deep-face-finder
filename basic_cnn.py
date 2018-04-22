# Adapted from https://www.tensorflow.org/tutorials/layers

from fruit_loader import fruit_loader
import tensorflow as tf

print("loading data")
test_images, test_labels = fruit_loader('test')
train_images, train_labels = fruit_loader('train')

#######################################
#####  Load in data  ##################
#######################################

#######################################
#####  Build neural net  ##############
#######################################


def next_batch(array, batch_size, index):
    length = len(array)
    if (index*batch_size)+batch_size >= length:
        index = (length/batch_size) % index
    batch = array[index*batch_size:(index*batch_size)+batch_size]
    return batch


print("initalizing network")

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 45, 45, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 60])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 60])
b_fc2 = bias_variable([60])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


batch_size = 50
num_epochs = 5

print ("starting train")
for epoch in range(num_epochs):
    for i in range(len(train_images) // batch_size):
    	batch = (next_batch(train_images, 50, i), next_batch(train_labels, 50, i))
    	if i%100 == 0:
    		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.95})
    		print("step %d, training accuracy %g"%(i, train_accuracy))
    	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("Epoch: %g"%epoch)
    print("test accuracy %g"%accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0}))

sess.close()
