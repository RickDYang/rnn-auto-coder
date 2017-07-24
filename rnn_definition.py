class rnn_definition:
    '''definition of a rnn'''
    def __init__(self, steps, hidden_size, LTSM_layers, keep_prob):
        '''constructor
        Keyword argument:
        steps -- steps size in rnn cell
        hidden_size -- feature numbers in hidden layer
        LTSM_layers -- LTSM layers number
        '''
        self.steps = steps
        self.hidden_size = hidden_size
        self.LTSM_layers = LTSM_layers
        self.keep_prob = keep_prob
