import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # intialize Z
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        
        #assign values to Z
        for b in range(batch_size):
            for k in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        A_slice = A[b, k, i:i + self.kernel, j:j + self.kernel]
                        Z[b, k, i, j] = np.max(A_slice)
        
        
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # intialize dLdA
        batch_size, out_channels, output_width, output_height = dLdZ.shape
    
        dLdA = np.zeros(self.A.shape)
        
        #find the max value in the A_slice and assign it to the corresponding position in dLdA
        for b in range(self.A.shape[0]):
            for k in range(self.A.shape[1]):
                for i in range(output_width):
                    for j in range(output_height):
                       #find the max value in the A_slice
                        A_slice = self.A[b, k, i:i + self.kernel, j:j + self.kernel]
                        max_val = np.max(A_slice)
                        #find the position of the max value
                        max_i, max_j = np.where(A_slice == max_val)
                        max_i, max_j = max_i[0], max_j[0]
                        #assign the value to the corresponding position in dLdA
                        dLdA[b, k, i:i + self.kernel , j:j + self.kernel][max_i,max_j] += dLdZ[b, k, i, j]

        return dLdA
        
        
    


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # intialize Z
        self.A =A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        
        #assign values to Z
        for i in range(output_height):
            for j in range(output_width):
                A_slice = A[:, :, i:i + self.kernel, j:j + self.kernel]
                Z[:, :, i, j] = np.mean(A_slice, axis=(2, 3))
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # intialize dLdA
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        dLdA = np.zeros(self.A.shape)
       
       
        #find the mean value in the A_slice and assign it to the corresponding position in dLdA
        for b in range(self.A.shape[0]):
            for k in range(self.A.shape[1]):
                for i in range(output_width):
                    for j in range(output_height):
                        #find the mean value in the A_slice
                        A_slice = self.A[b, k, i:i + self.kernel, j:j + self.kernel]
                        mean_val = np.mean(A_slice)
                        #assign the value to the corresponding position in dLdA
                        dLdA[b, k, i:i + self.kernel, j:j + self.kernel] += dLdZ[b, k, i, j]/(self.kernel**2)
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        #call maxpool2d_stride1
        Z = self.maxpool2d_stride1.forward(A)
        #downsample
        Z = self.downsample2d.forward(Z)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        #upsample dLdZ
        dLdZ = self.downsample2d.backward(dLdZ)
        #backward pass through maxpool2d_stride1
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        #call meanpool2d_stride1
        Z = self.meanpool2d_stride1.forward(A)
        #downsample
        Z = self.downsample2d.forward(Z)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        #upsample dLdZ
        dLdZ = self.downsample2d.backward(dLdZ)
        #backward pass through meanpool2d_stride1
        dLdA = self.meanpool2d_stride1.backward(dLdZ)
        
        return dLdA
