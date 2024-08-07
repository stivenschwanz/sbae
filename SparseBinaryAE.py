import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TopKBits(nn.Module):
    """
    Extends the nn.Module class to implement a top-k bits activation.
    """
    def __init__(self, k=10):
        """
        Default constructor.

        Parameters:
            k (int): number of activated bits
        """
        super(TopKBits, self).__init__()
        
        self.k = k
        
    def forward(self, x):
        """
        Forward function defining the activation function computation.

        Parameters:
            x (float[]): dense input vector
        
        Returns:
            (float[]): sparse binary output vector
        """
        
        # Find the indices of the k largest values 
        _, k = torch.topk(x, self.k)
        
        # Build a k-sparse binary vector
        b = torch.zeros(x.shape)        
        r, c = k.shape
        idx0 = torch.arange(r).reshape(-1, 1).repeat(1, c).flatten()
        idx1 = k.flatten()
        b[idx0, idx1] = 1

        return b
        
        
class SparseBinaryAE(nn.Module):
    """
    Extends the nn.Module class to implement a Sparse Binary Autoencoder.
    
    Adapted from https://www.cs.toronto.edu/%7Elczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, input_shape=(1,28,28), latent_size=1024, activated_bits=32, loss_fn=F.mse_loss, lr=1e-3, l2=1e-5):
        """
        Default constructor.

        Parameters:
            input_shape (int,int,int): input vector shape (channels, width, height)
            latent_size (int): latent space size
            activated_bits (int): number of activated bits
            loss_fn (function): loss function
            lr (float): learning rate
            l2 (float): l2 penalty (weight decay)
        """
        super(SparseBinaryAE, self).__init__()
        
        # Store input parameters
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.loss_fn = loss_fn
        self._loss = None
                
        # Compute the shape before flattening
        shape_before_flattening = (64, input_shape[1] // 28, input_shape[2] // 28)
        
        # Compute the flattened size after convolutions
        flattened_size = 64 * (input_shape[1] // 28) * (input_shape[2] // 28)
        
        # Build the convolutional encoder        
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(input_shape[0], 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(flattened_size, latent_size),
            nn.Sigmoid(),
            TopKBits(activated_bits)
        )
        
        # Build the convolutional decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, flattened_size),
            nn.Unflatten(1, shape_before_flattening),
            nn.ConvTranspose2d(64, 32, 7, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_shape[0], 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
        # Configure the optimizer
        self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

    def forward(self, x):
        """
        Forward function defining the encoder computation.

        Parameters:
            x (float[]): dense input vector
        
        Returns:
            (float[]): dense output vector
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        """
        Encode function.

        Parameters:
            x (float[]): dense input vector
        
        Returns:
            (float[]): sparse binary output vector
        """
        
        # View the input data as a contiguous vector
        #x = x.view(-1, self.input_size)
        
        # Encode the data as a sparse binary latent representation
        with torch.no_grad():
            return self.encoder(x)
    
    def decode(self, h):
        """
        Decode function.

        Parameters:
            x (float[]): sparse binary input vector
        
        Returns:
            (float[]): dense output vector
        """
        
        # Decode the latent representation
        with torch.no_grad():
            return self.decoder(h)
    
    def loss(self, x, target, **kwargs):
        """
        Loss function.

        Parameters:
            x (float[]): dense input vector
            target (float[]): dense target vector
        
        Returns:
            (float): loss function value
        """
        
        # Compute the loss function
        self._loss = self.loss_fn(x, target, **kwargs)
        
        return self._loss