import pickle
import numpy as np

def softmax(x):
        xt = np.exp(x - np.max(x))
        return xt/np.sum(xt)

class RnnNumpy:
    
    #Constructor
    def __init__(self, word_dim, hidden_neuron=None, bptt_truncate=None):
        
        self.__hidden_neuron = hidden_neuron if hidden_neuron is not None else 100
        self.__bptt_truncate = bptt_truncate if bptt_truncate is not None else 4
        self.__word_dim = word_dim
        
        self.__U = np.random.uniform(-np.sqrt(1./self.__word_dim), np.sqrt(1./self.__word_dim), (self.__hidden_neuron, self.__word_dim))
        self.__V = np.random.uniform(-np.sqrt(1./self.__hidden_neuron), np.sqrt(1./self.__hidden_neuron), (self.__word_dim, self.__hidden_neuron))
        self.__W = np.random.uniform(-np.sqrt(1./self.__hidden_neuron), np.sqrt(1./self.__hidden_neuron), (self.__hidden_neuron, self.__hidden_neuron))
        self.__deltaU = np.random.uniform(-np.sqrt(1./self.__word_dim), np.sqrt(1./self.__word_dim), (self.__hidden_neuron, self.__word_dim))
        self.__deltaV = np.random.uniform(-np.sqrt(1./self.__hidden_neuron), np.sqrt(1./self.__hidden_neuron), (self.__word_dim, self.__hidden_neuron))
        self.__deltaW = np.random.uniform(-np.sqrt(1./self.__hidden_neuron), np.sqrt(1./self.__hidden_neuron), (self.__hidden_neuron, self.__hidden_neuron))
    
    def forwardPass(self, X):
        # X = [x_{0}, ..., x_{t}, ... x{T-1}]: Input Sequence
        # S = [S_{0}^{0}, ...;... S_{t}^{m}, ...; .....S{T}^{H}: Hidden States at t and neuron m
        # S = R^{(T+1), H}
        # O = [O_{0}, ...O_{t}, ... O_{T-1}]: Output Sequence
        
        T = len(X)
        S = np.zeros((T+1, self.__hidden_neuron));
        S[-1] = np.zeros(self.__hidden_neuron);
        O = np.zeros((T, self.__word_dim))

        for t in np.arange(T):
            inV = self.__U[:, X[t]]
            S[t] = np.tanh(self.__U[:, X[t]] + self.__W.dot(S[t-1]))
            O[t] = softmax(self.__V.dot(S[t]))

        return [O, S]
    
    def bptt(self, X):
        O,S = self.forwardPass(X)
        
        for t in np.arange(len(X))[::-1]:
            self.__deltaU += 



    def predict(self, x):
        O, S = self.forwardPass(x)
        return np.argmax(O, axis=1)

    def errFunc(self, X, Y):
       
        T = len(X)
        E = 0
        P = self.predict(X)
        for t in range(T):
            E += Y[t]*np.log2(P[t])

        return E/T

if __name__ == "__main__":
    
    with open('X_train.pk','rb') as f:
        X_train = pickle.load(f)

    with open('Y_train.pk', 'rb') as f:
        Y_train = pickle.load(f)

    np.random.seed(10)
    RnnModel = RnnNumpy(8000, 100, 4)
    O, S = RnnModel.forwardPass(X_train[10])
    print O
    print RnnModel.predict(X_train[10])
    err = RnnModel.errFunc(X_train[10], Y_train[10])
    print err
