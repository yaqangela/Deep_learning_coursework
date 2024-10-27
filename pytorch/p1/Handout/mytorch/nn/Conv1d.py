# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(output_size):
            A_slice = A[:, :, i:i + self.kernel_size]  # (batch_size, in_channels, kernel_size)
            
            Z[:, :, i] = np.tensordot(A_slice, self.W, axes=([1, 2], [1, 2])) + self.b  # (batch_size, out_channels)
    
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        batch_size, out_channels, output_size = dLdZ.shape
        input_size = self.A.shape[2]
        dLdA = np.zeros((batch_size, self.in_channels, input_size))
        
        # iterate over output_size
        for i in range(self.kernel_size): 
            A_slice = self.A[:, :, i:i + output_size] # (batch_size, in_channels, output_size) 
            #
            self.dLdW[:,:,i] += np.tensordot(dLdZ, A_slice, axes=((0,2),(0,2))) # (out_channels, in_channels, kernel_size)
            
        self.dLdb += np.sum(dLdZ, axis=(0, 2))  # (out_channels)
        
        #pad dLdZ with kernal_size -1
        dLdZ_padded = np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1, self.kernel_size-1)))# (batch_size, out_channels, input_size)
        
        # Flip the weights along the kernel axis (axis=-1)
        W_flipped = np.flip(self.W, axis=-1) # (out_channels, in_channels, kernel_size)
        
        #Convolve each flipped channel of the filter with the broadcasted and padded dLdZ to get dLdA
        for i in range(input_size):
            # Extract the slice of dLdZ with the same width as the kernel size
            dLdZ_slice = dLdZ_padded[:, :, i:i + self.kernel_size]  # Shape: (batch_size, out_channels, kernel_size)
        
        # Apply the convolution for each output channel
            for out_c in range(out_channels):
                #  apply the convolution for each input channel
                for in_c in range(self.in_channels):
                    dLdA[:, in_c, i] += np.sum(dLdZ_slice[:, out_c, :] * W_flipped[out_c, in_c, :], axis=-1)

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)


    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        # Get the input shape
        batch_size, in_channels, input_size = A.shape

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0,0), (0,0), (self.pad, self.pad)))

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #get the shape of dLdZ
        batch_size, out_channels, output_size = dLdZ.shape
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad]

        return dLdA
