import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import os
from tensorflow.python.keras import backend as K
inception_feat = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=True)
import matplotlib.pyplot as plt


n_nodes_hl1 = 512
n_nodes_hl2 = 256
n_nodes_hl3 = 128
n_nodes_hl4 = 32
n_nodes_hl5 = 8

batch_size = 100

loss_list = []
n_classes = 1
keep_prob = 0.2
x = tf.placeholder('float32',[None,2048],name = 'inputs')
y = tf.placeholder(tf.float32,name= 'pd')
# normal.txt is the text file containing normalised dankness
num_lines = len(open('Normal.txt', "r").readlines())
idx=0
mega_feat = np.random.rand(num_lines, 2048)
mega_feat = mega_feat/mega_feat.sum(axis=1)[:,None]
train_y = np.loadtxt('Normal.txt', dtype= 'float32', comments='#', delimiter='\n', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes')
train_x = mega_feat
print(mega_feat)
def neural_network_model(data):
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([2048, n_nodes_hl1])), 'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}

    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])), 'biases' : tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.dropout(l1, keep_prob)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.dropout(l2, keep_prob)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    l3 = tf.nn.dropout(l3, keep_prob)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    l4 = tf.nn.dropout(l4, keep_prob)

    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.relu(l5)
    l5 = tf.nn.dropout(l5, keep_prob)

    output = tf.matmul(l5,output_layer['weights']) + output_layer['biases']

    return output

hm_epochs = 10
saver = tf.train.Saver()
def train_neural_network(x):
    prediction = neural_network_model(x)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    cost = tf.losses.mean_squared_error(y,prediction)
    #learning rate = 0.01
    optimizer = tf.train.AdamOptimizer().minimize(cost)



    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < num_lines :
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                loss_list.append(c)
                plt.plot(loss_list)
                epoch_loss += c
                i += batch_size
            print('epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        #saver.save(sess, r'\H:\MEGH\NITK\Third Year - B.Tech NITK\DankNotDank\src\my_test_model.txt')

            # Saving
        inputs = {"inputs": x}
        outputs = {"pd": y}
        tf.saved_model.simple_save(sess, r'H:\MEGH\NITK\Third Year - B.Tech NITK\DankNotDank\src\new', inputs, outputs)

train_neural_network(x)
plt.xlabel('# of epochs')
plt.ylabel('Loss')
plt.xlim((0,hm_epochs))
plt.show()
#this is the code for training our model
# we will do 10 epochs for demo purposes but the actual model has been trained for 1000 epochs
# as we can see the value of loss function is decreasing