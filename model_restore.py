import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
graph = tf.Graph()
def prd(im):
    cv2.imshow('image', im)
    a = np.sum(np.sum(im, axis=1), axis=0) / 10000
    print(np.ceil(a))
    cv2.waitKey()
with graph.as_default():
    with tf.Session() as sess:

        #floating = tf.Variable(3.14159265359, tf.float32)
        #arr = floating.eval()
        x_input = np.random.rand(1, 2048)
        x_input = x_input / x_input.sum(axis=1)[:, None]
        #tf.initializers.variables(
          #floating, x_input)
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
        r'H:\MEGH\NITK\Third Year - B.Tech NITK\DankNotDank\src\new',
        )
        x = graph.get_tensor_by_name('inputs:0')
        #prediction = graph.get_tensor_by_name('pd:0')

        #sess.run(prediction, feed_dict={x: x_input})
        #kk = prediction.eval()
        #print(prediction.eval())



img = cv2.imread(r"H:\MEGH\NITK\Third Year - B.Tech NITK\DankNotDank\test1.png",0)

#test our model on 1 or 2 images
#prediction highlighted
prd(img)
