import os
import sys
import time
from auto_coder_task import *
from rnn_definition import *
from linux_code import *
from training_parameters import *
import code_data
import tensorflow as tf

# rnn model definitions
model_def = rnn_definition(steps = 32, hidden_size = 512, LTSM_layers = 4)

# for debug
#parameters = train_parameters(learning_rate = 0.001, batch_size = 10,
#    max_iterations = 100, display_step= 10)

# training paramaters
parameters = train_parameters(learning_rate = 0.001, batch_size = 500,
    max_iterations = 50000, display_step= 1000)

# source data
data = linux_code('/home/*/linux-source-4.4.0',['.c'])

# where to store input texts to generate in which each line presents a prefix to generate
generate_input = 'input_prefix.txt'
# where to store output results 
generate_output = 'code_gen.txt'
# how many charaters to generate
generate_size = 200

def main():
    #path = train()
    # need to reset otherwise new nn will fail to create
    #tf.reset_default_graph()
    generate('201706010910')

def train():
    '''train this model
    '''
    task = auto_coder_task(model_def)

    print('start training', time.strftime('%Y-%m-%-d-%H:%M.%S'))
    
    path = task.train(data, parameters)

    print('training completed', time.strftime('%Y-%m-%d-%H:%M.%S'))
    return path

def generate(path = None):
    '''generate output by current training result
    Keyword argument:
    path -- path to store training result
    '''
    # get path from input, or from argv, or the first sub dir
    if path == None:
        if len(sys.argv) > 1:
            path = sys.argv[1]
        else:
            #choose the first sub dir
            cur = os.getcwd();
            dirs = [d for d in os.listdir(cur) if os.path.isdir(os.path.join(cur,d))]
            path = dirs[0]

    task = auto_coder_task(model_def)
    task.generate(path, generate_callback)

def generate_callback(session, model):
    '''callback function to generate somethin
    Keyword argument:
    session -- tensorflow session which has been restored
    model -- the tensorflow model to calculate predictions
    '''
    text = ''
    with open(generate_input) as f:
        for line in f:
            text += line

            if (len(text) >= model_def.steps):
                generate_one(session, model, text, generate_size)
                text = ''

    print('Generating completed')

def generate_one(session, model, prefix, length):
    '''Generate one text session from trained model
    Keyword argument:
    session -- tensorflow session which has been restored
    model -- the tensorflow model to calculate predictions
    prefix -- prefix string to start generating
    length -- length to generate
    '''
    prefix = prefix[:model_def.steps]
    res = prefix[:]
    batch_y = numpy.zeros([1, code_data.vocabulary_size], dtype = int)

    for i in range(length):
        # output is steps x n size 
        batch_x = code_data.str2input(prefix)
        # need to reshape to 1 x steps x n size
        batch_x = numpy.reshape(batch_x, [1, model_def.steps, code_data.vocabulary_size])
        # get prediction
        pred_res = session.run(model.pred, feed_dict={model.x:batch_x, model.y:batch_y})

        # get max element's index in prediction
        pred_char = code_data.id2char(numpy.argmax(pred_res))
        
        res += pred_char
        # move to next text frame
        prefix = prefix[1:] + pred_char

    # ouput result
    with open(generate_output, 'a') as f:
        f.writelines('########New#########\n')
        f.writelines(res)
        f.writelines('\n\n')

    return res


if __name__ == '__main__':
    main()