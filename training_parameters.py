class train_parameters:
    def __init__(self, learning_rate, batch_size, max_iterations, display_step=1000):
        '''constructor
        Keyword argument:
        learning_rate -- the alpha learning rating in gradient descent
        batch_size -- data size in mini-batch regression
        max_iterations -- max iteration to train
        display_step -- the interval to display training status.
        '''
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.display_step = display_step