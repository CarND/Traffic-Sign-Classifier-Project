import tensorflow.compat.v1 as tf
import numpy as np 

class Sharpen(tf.keras.layers.Layer):
    """ sharpen edges of image """
    def __init__(self, num_outputs):
        super(Sharpen, self).__init__()
        self.num_outputs = num_outputs
    def build(self, input_shape):
        self.kernel = np.array(([-1,-1,-1], [-1,9,-1], [-1,-1,-1]))
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.cast(self.kernel, tf.float32)
    def call(self, input_):
        return tf.nn.conv2d(input_, self.kernel, strides=[1,1,1,1], padding='SAME')        