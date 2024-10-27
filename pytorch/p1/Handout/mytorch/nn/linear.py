import numpy as np


class Linear:

    def __init__(self, in_features, out_features, weight_init_fn,bias_init_fn, debug=True):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features,in_features))  # (c1, c0)
        self.b = np.zeros((out_features,1))  #(c1, 1)

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  #  shape (N, C0)
        self.N = A.shape[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = A.dot(self.W.T) + self.Ones.dot(self.b.T)  # (N, C1)

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ.dot(self.W)  #(N, C0) * (C1, C0) = (N, C1)
        self.dLdW = dLdZ.T.dot(self.A)  # TODO
        self.dLdb = dLdZ.T.dot(np.ones((self.N,1)))  # TODO

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
