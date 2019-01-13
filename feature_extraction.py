#feature vectors extracted using google collaboratory online GPU
import tensorflow as tf
import os
import sys
from PIL import Image
import numpy as np
import tensorflow_hub as hub
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config = config)
idx = 0
#jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000
mega_feat = np.zeros([3338, 2048], dtype='float32')
inception_feat = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with open('/content/gdrive/My Drive/foo.txt', 'a') as f:
      for fname in os.listdir('/content/gdrive/My Drive/Memes/image1') :
          path = '/content/gdrive/My Drive/Memes/image1/' + fname
          img = Image.open(path)
          np_img = np.array(img, dtype=np.float32) / 255.0
          batch = np.array([np_img])
          feat = inception_feat(batch)
          #np.save('Inception_V3_feature_vectors.npy',mega_feat, allow_pickle=True, fix_imports=True)
          array = feat.eval()
          mega_feat[idx,:] = array
          b = array.tolist()
          f.write(str(b) + '\n')
          idx += 1
          print(idx)
f.close()