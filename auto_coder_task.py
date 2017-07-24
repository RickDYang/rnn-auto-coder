import tensorflow as tf
import time
import os
import logging
from tensorflow.contrib import rnn
from rnn_model import *
from code_data import *

class auto_coder_task():

    # file name to save/restor trained model
    __model_path = 'auto_code_model.ckpt'
    # log file name
    __training_log = 'train.log'

    '''define a task for auto code'''
    def __init__(self, model_def):
        '''constructor
        Keyword argument:
            model_def -- model defintion for rnn
        '''
        self.model_def = model_def

    def train(self, data, paras, generate_callback):
        '''train an rnn model and return the path where the session stored
        Keyword argument:
        data -- trainig data
        paras -- training parameters
        '''
        model = rnn_model(self.model_def)

        log_file = time.strftime('%Y%m%d%H%M_') + auto_coder_task.__training_log
        logging.basicConfig(filename=log_file, level=logging.INFO)

        #with tf.Session(config=config_noGPU) as session:
        with tf.Session() as session:
            # define gradient descent optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=paras.learning_rate).minimize(model.cost)

            # initialize the variables
            initializer = tf.global_variables_initializer()
            session.run(initializer)

            iteration = 1
            # get validation data
            batch_v = data.validation_data.get_cache_batch(paras.batch_size * 10, self.model_def.steps)
            
            try:
                # keep training until reach max iterations
                while  iteration <= paras.max_iterations:
                    batch = data.training_data.next_batch(paras.batch_size, self.model_def.steps)
                    
                    # reach end of data is returned batch size less than required batch size
                    end = batch[0].shape[0] < paras.batch_size

                    session.run(optimizer, feed_dict={model.x:batch[0], model.y:batch[1]})

                    # display traing status
                    if iteration % paras.display_step == 0 or end:
                        status = self.cal_status(session, model, iteration, batch, batch_v)
                        print(status)
                        logging.info(status)

                    if end:
                        print("Iteration {} No more data".format(iteration))
                        break

                    iteration += 1
            finally:
                if generate_callback is not None:
                    generate_callback(session, model)

                # save session model & data for later loading and generate
                path = self.__save_training(session)

                print("training complete", path)
                return path

    def cal_status(self, session, model, iteration, batch, batch_v):
        # status for current training data
        acc, loss = session.run([model.accuracy, model.cost], feed_dict={model.x:batch[0], model.y:batch[1]})
        # status for validation data
        acc_v, loss_v = session.run([model.accuracy, model.cost], feed_dict={model.x:batch_v[0], model.y:batch_v[1]})
        return "{} Iteration {} Training/Validation Set - Loss={:.6f} / {:.6f}, Accuracy={:.5f} / {:.5f} "\
            .format(time.strftime('%Y-%m-%-d-%H:%M.%S'), iteration, loss, loss_v, acc, acc_v)


    def generate(self, cur_dir, generate_callback):
        '''generate code from trained model
        Keyword argument:
        dir_path to 
        cur_dir --path to dir where training result saved
        generate_callback -- callback function to generate by caller
        '''
        model = rnn_model(self.model_def)

        session_path = os.path.join(cur_dir, auto_coder_task.__model_path)

        saver = tf.train.Saver()
        with tf.Session() as session:
            #restore tensorflow session from saved folder
            saver.restore(session, session_path)
            # callback function
            generate_callback(session, model)

    def __save_training(self, session):
        '''return dir path where current training results saved
        Keyword argument:
        session -- the tensorflow session which have trained results to save
         '''
        cur_dir = time.strftime('%Y%m%d%H%M')
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

        tf.train.Saver().save(session, os.path.join(cur_dir, auto_coder_task.__model_path))

        return cur_dir