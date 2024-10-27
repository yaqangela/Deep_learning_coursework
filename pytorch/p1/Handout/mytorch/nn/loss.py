import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = np.shape(A)[0]  # TODO
        self.C = np.shape(A)[1]  # TODO
        se = (A - Y)**2 # TODO

        sse = np.sum(se) # TODO
        
        mse = sse/(self.N * self.C)  # TODO

        return mse
    
    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C) 

        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = np.shape(A)[0]
        self.C = np.shape(A)[1]

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        A = A - np.max(A, axis=1, keepdims=True)
        EXP_A = np.exp(A)
        self.softmax = EXP_A / np.sum(EXP_A, axis=1,keepdims=True)  
        cross_entropy = (-Y * np.log(self.softmax)).dot(Ones_C)
        sum_crossentropy =np.sum(cross_entropy)
        L = sum_crossentropy / self.N

        return L

    def backward(self):
        dLdA = (self.softmax - self.Y) / self.N

        return dLdA
