class RnnNumpy:
    
    #Constructor
    def __init__(self, word_dim, hidden_neuron=None, bptt_truncate=None):

        self.__hidden_neuron = hidden_neuron if hidden_neuron is not None else 100
        self.__bptt_truncate = bptt_truncate if bptt_truncate is not None else 4
        self.__word_dim = word_dim
        self.__U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.__V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.__W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forwardPass(self, inputFeat):
        #X = 1 x len(inputFeat); 1 x 8000 in this example
        #H = 1 x 100
        #h_{t} = f(WX + Uh_{t-1})
        #W = 100 x 8000, U = 100 x 100
