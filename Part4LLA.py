from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# mnist = fashion_mnist.read_data_sets()

data = input_data.read_data_sets('data/fashion',
                                 source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)

lr = 0.001 #change to adaptive
num_steps = 1200
batch_size = 1000
display_step = 60

#parameter
num_input = 784
num_classes = 10
dropout = 1.0
lossarr = []
accarr = []

#graph input
X = tf.placeholder(tf.float32, [None, num_input])
print(X)
Y = tf.placeholder(tf.float32, [None, num_classes])
prob = tf.placeholder(tf.float32)

#define layer methods

def conv2dp(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv2dnp(x, W, b,strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1,k,k,1],
                          padding='SAME')


#model
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    #Convolution_1
    conv1 = conv2dnp(x, weights['wc1'], biases['bc1'])
    #Pool_1
    conv1 = maxpool2d(conv1, k=2)

    #Convolution_2
    conv2 = conv2dp(conv1, weights['wc2'], biases['bc2'])
    #Pool_2
    conv2 = maxpool2d(conv2, k=2)

    #FCL_1
    fc1 = tf.reshape(conv2, [-1, weights['wfc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    #FCL_2
    # fc2 = tf.reshape(fc1, [-1, weights['wfc2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc1, weights['wfc2']), biases['bfc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # #FCL_3
    # fc3 = tf.reshape(fc2, [-1, 50])
    # fc3 = tf.add(tf.matmul(fc3, weights['wfc3']), biases['bfc3'])
    # fc3 = tf.nn.softmax(fc3)
    # fc3 = tf.nn.dropout(fc3, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

weights = {
    # 5x5 conv, 1 input, 3 outputs
    'wc1': tf.get_variable('WC1', shape = (5, 5, 1, 3), initializer = tf.contrib.layers.xavier_initializer()),
    # 3x3 conv, 3 inputs, 3 outputs
    'wc2': tf.get_variable('WC2', shape = (3, 3, 3, 3), initializer = tf.contrib.layers.xavier_initializer()),
    # fully connected, 108 inputs, 100 outputs
    'wfc1': tf.get_variable('WFC1', shape=(108, 100), initializer= tf.contrib.layers.xavier_initializer()),
    # fully connected, 100 inputs, 50 outputs
    'wfc2': tf.get_variable('WFC2', shape=(100, 50), initializer= tf.contrib.layers.xavier_initializer()),
    # fully connected, 50 inputs, 10 outputs
    # 'wfc3': tf.Variable(tf.random_normal([50, 10])),
    # 10 inputs, 10 outputs (class prediction)
    'out': tf.get_variable('WOUT', shape = (50, num_classes)),
}

biases = {
    'bc1': tf.get_variable('BC1', shape = (3), initializer= tf.zeros_initializer()),
    'bc2': tf.get_variable('BC2', shape = (3), initializer= tf.zeros_initializer()),
    'bfc1': tf.get_variable('BFC1', shape = (100), initializer= tf.zeros_initializer()),
    'bfc2': tf.get_variable('BFC2', shape = (50), initializer= tf.zeros_initializer()),
    # 'bfc3': tf.Variable(tf.random_normal([10])),
    'out': tf.get_variable('BOUT', shape = (num_classes), initializer= tf.zeros_initializer()),
}
# Construct model
logits = conv_net(X, weights, biases, prob)
logits = tf.nn.dropout(logits, dropout)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        batch_x = batch_x.reshape(-1, num_input)
        batch_y = batch_y.reshape(-1, num_classes)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X:batch_x, Y: batch_y, prob: dropout})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                prob: 1 })
            lossarr.append(loss)
            accarr.append(acc)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 10000 Fashion MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: data.test.images[:10000],
                                      Y: data.test.labels[:10000],
                                      prob: 1}))
    #saving model over here
    # savedmodel = tf.train.Saver()
    # savedmodel.save(sess, './model.ckpt')

    plt.plot(lossarr)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(accarr)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


# print(graphing.history.keys())
# # summarize graphing for accuracy
# # plt.plot(graphing.history['loss_op'])
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
# # plt.show()
#  summarize graphing for loss
# plt.plot('loss_op')
# # plt.plot(graphing.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#gittutend
