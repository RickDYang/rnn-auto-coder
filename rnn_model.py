import tensorflow as tf
from tensorflow.contrib import rnn
import code_data

class rnn_model:
    '''define a cnn model'''
    def __init__(self, model_def):
        '''construct a model
        
        Keyword argument:
        model_def -- model definition of this rnn
        '''

        def create_lstm(model_def):
            # define LSTM cell
            cell = rnn.BasicLSTMCell(model_def.hidden_size,
                forget_bias=0.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
            # process dropout
            if model_def.keep_prob < 1 :
                 cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=model_def.keep_prob)

            return cell

        # define placeholder for input:x and output:y
        # input x has ? x steps x n size
        # where each charater is represented as (0,.., 0, 1, 0, ... 0) form
        self.x = tf.placeholder(tf.float32, [None, model_def.steps, code_data.vocabulary_size])
        # output y has ? x n size 
        # where each charater is represented as (0,.., 0, 1, 0, ... 0) form
        self.y = tf.placeholder(tf.float32, [None, code_data.vocabulary_size])
        
        # define weights between hidden layer to output layer
        weights = {
            'out': tf.Variable(tf.random_normal([model_def.hidden_size, code_data.vocabulary_size]))
        }

        # define bias from hidden layer to output layer
        biases = {
            'out': tf.Variable(tf.random_normal([code_data.vocabulary_size]))
        }

        #with tf.device("/cpu:0" ):
        # multiple LSTM layers
        lstm_cell = rnn.MultiRNNCell([create_lstm(model_def) for _ in range(model_def.LTSM_layers)],
                state_is_tuple=True)

        # get LSTM cell output
        # need to unstack x by steps twith tf.device('/cpu:0'):o be 1 x n vector which presents a charater
        _x = tf.unstack(self.x, model_def.steps, 1)
        outputs, states = rnn.static_rnn(lstm_cell, _x, dtype=tf.float32)

        # define output prediction. z = x*w + b
        # predition is to get index of max value of output, 
        # so we do not need to apply sigmoid for prediction
        self.pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

        # use softmax_cross_entropy as cost funtion
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
 
        # define accuracy to evaludate model
        # to compare indexes of max value
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

