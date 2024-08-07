import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    """
    Extends the nn.Module class to implement a sparse binary encoder.
    """
    def __init__(self, input_shape=(1,28,28), latent_size=1024, activated_bits=10):
        """
        Default constructor.

        Parameters:
            input_shape (int,int,int): input vector shape (channels, width, height)
            latent_size (int): latent space size
            activated_bits (int): number of activated bits
        """
        
        super(Encoder, self).__init__()
        
         # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Compute the flattened size after convolutions
        flattened_size = 128 * (input_shape[1] // 4) * (input_shape[2] // 4)
        
        # Define the fully connected layer to create embeddings
        self.fc1 = nn.Linear(flattened_size, latent_size)        
        
        self.k = activated_bits
    
    def forward(self, x):
        """
        Forward function defining the encoder computation.

        Parameters:
            x (float[]): dense input vector
        
        Returns:
            (float[]): sparse binary output vector
        """
        
        # Apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Apply the fully connected layer to generate embeddings
        x = self.fc1(x)
                
        # Apply the sigmoid activation function
        y = torch.sigmoid(x)
        
        # Find the indices of the k largest values 
        _, k = torch.topk(y, self.k)
        
        # Build a k-sparse binary vector
        b = torch.zeros(y.shape)        
        r, c = k.shape
        idx0 = torch.arange(r).reshape(-1, 1).repeat(1, c).flatten()
        idx1 = k.flatten()
        b[idx0, idx1] = 1

        return b
    
class Decoder(nn.Module):
    """
    Extends the nn.Module class to implement a binary decoder.
    """
    def __init__(self, latent_size=1024, output_shape=(1,28,28)):
        """
        Default constructor.

        Parameters:
            latent_size (int): latent space size
            output_shape (int,int,int): output vector shape (channels, width, height)
        """
         
        super(Decoder, self).__init__()
        
        # Store the shape before flattening
        self.reshape_dim = (128, output_shape[1] // 4, output_shape[2] // 4)
        
        # Compute the flattened size after convolutions
        flattened_size = 128 * (output_shape[1] // 4) * (output_shape[2] // 4)
        
        # Define the fully connected layer to unflatten the embeddings
        self.fc1 = nn.Linear(latent_size, flattened_size)
        
        # Define the transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0)
        
        # Define the final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, output_shape[0], kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        """
        Forward function defining the decoder computation.

        Parameters:
            x (float[]): sparse binary input vector
        
        Returns:
            (float[]): dense output vector
        """
        
        # Apply the fully connected layer to unflatten the embeddings
        x = self.fc1(x)
        
        # Reshape the tensor to match output shape (before flattening)
        x = x.view(x.size(0), *self.reshape_dim)
        
        # Apply the ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        
        # Apply the sigmoid activation to the final convolutional layer to generate output image
        y = torch.tanh(self.conv1(x))
        
        return y
    
class Sbae(nn.Module):
    """
    Extends the nn.Module class to implement a Sparse Binary Autoencoder.
    """
        
    def __init__(self, input_shape=(1,28,28), latent_size=10, activated_bits=10, loss_fn=F.l1_loss, lr=1e-4, l2=0.):
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
         
        super(Sbae, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.E = Encoder(input_shape, latent_size, activated_bits)
        self.D = Decoder(latent_size, input_shape)
        self.loss_fn = loss_fn
        self._loss = None
        self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        
    def forward(self, x):
        """
        Forward function defining the autoencoder computation.

        Parameters:
            x (float[]): dense input vector
        
        Returns:
            (float[]): dense output vector
        """
        
        # View the input data as a contiguous vector
        #x = x.view(-1, self.input_size)
        
        # Encode the data as a sparse binary latent representation
        h = self.E(x)
        
        # Decode the latent representation
        out = self.D(h)
        
        return out
    
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
            return self.E(x)
    
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
            return self.D(h)
    
    def loss(self, x, target, **kwargs):
        """
        Loss function.

        Parameters:
            x (float[]): dense input vector
            target (float[]): dense target vector
        
        Returns:
            (float): loss function value
        """
        
        # View the target data as a contiguous vector
        #target = target.view(-1, self.input_size)
        
        # Compute the loss function
        self._loss = self.loss_fn(x, target, **kwargs)
        
        return self._loss
