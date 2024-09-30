import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        W_out =(A.shape[2]-1)*self.upsampling_factor + 1
        # initialize Z with zeros
        Z = np.zeros(A.shape[0],A.shape[1],W_out)
        # fill in the values
        Z[:,:,::self.upsampling_factor] = A
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # fill in the values
        dLdA = dLdZ[:,:,::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        Z = A[:,:,::self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        W_in = dLdZ.shape[2]*self.downsampling_factor - 1
        dLdA = np.zeros(dLdZ.shape[0],dLdZ.shape[1],W_in)
        dLdA = dLdZ[:,:,::self.downsampling_factor]
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        H_out = (A.shape[2]-1)*self.upsampling_factor + 1
        W_out = (A.shape[3]-1)*self.upsampling_factor + 1
        Z = np.zeros(A.shape[0],A.shape[1],H_out,W_out)
        Z[:,:,::self.upsampling_factor,::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        Z = A[:,:,::self.downsampling_factor,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        H_in = dLdZ.shape[2]*self.downsampling_factor - 1
        W_in = dLdZ.shape[3]*self.downsampling_factor - 1
        dLdA = np.zeros(dLdZ.shape[0],dLdZ.shape[1],H_in,W_in)
        dLdA = dLdZ[:,:,::self.downsampling_factor,::self.downsampling_factor]
        return dLdA
