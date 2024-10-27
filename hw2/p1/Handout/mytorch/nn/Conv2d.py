import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
            Weights (np.array): (out_channels, in_channels, kernel_size, kernel_size)
        """
        self.A = A
        
        # initialize Z
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))
        # apply convolution
        for b in range(batch_size):
            for k in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        Z[b, k, i, j] = np.sum(A[b, :, i:i + self.kernel_size, j:j + self.kernel_size] * self.W[k]) + self.b[k]
                                
                
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
       #pad dLdZ map with k-1 zeros
        dLdZ_padded = np.pad(dLdZ, ((0,0), (0,0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)), 'constant')
        #flip the filter top to bottom and left to right
        W_flipped = np.flip(self.W, (2, 3))
        #initialize dLdA
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        input_height = output_height + self.kernel_size - 1
        input_width = output_width + self.kernel_size - 1
        dLdA = np.zeros((batch_size, self.in_channels, input_height, input_width))
           # Calculate dLdA
        for i in range(input_height):
            for j in range(input_width):
                # Extract patch of dLdZ_padded (padded gradient of Z)
                dLdZ_slice = dLdZ_padded[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                # Convolve dLdZ_slice with flipped filters to get dLdA
                for i_c in range(self.in_channels):
                    for o_c in range(out_channels):
                        dLdA[:, i_c, i, j] += np.sum(dLdZ_slice[:, o_c] * W_flipped[o_c, i_c], axis=(1, 2))

        #get dLdW
        # Calculate dLdW (gradient w.r.t. weights)
        for i in range(output_height):
            for j in range(output_width):
                A_slice = self.A[:, :, i:i + self.kernel_size, j:j + self.kernel_size] # (batch_size, in_channels, kernel_size, kernel_size)
                for out_c in range(out_channels):
                    # dLdZ[:, out_c, i, j]: shape (batch_size,)
                    self.dLdW[out_c] += np.tensordot(dLdZ[:, out_c, i, j], A_slice, axes=([0], [0]))
    
        #get dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA
