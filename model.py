import torch
import numpy as np
import torch.nn as nn
from gym import spaces
from torch import flatten
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, MaxPool2d, Module, ReLU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):
  """
  A basic implementation of a Deep Q-Network.
  """
  def __init__(self, action_space: spaces.Discrete):
    """
    Initialise the DQN
    :param action_space: the action space of the environment
    """
    super().__init__()
    
    # 1. initialize first set of CONV => RELU => POOL layers
    self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5)).to(device)
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    # 2. initialize second set of CONV => RELU => POOL layers
    self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)).to(device)
    self.relu2 = ReLU()
    self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    # 3. initialize first set of FC => RELU layers
    self.fc1 = Linear(in_features=1600, out_features=500).to(device)
    self.relu3 = ReLU()
    self.fc2 = Linear(in_features=500, out_features=action_space.n).to(device)

  def forward(self, x):
    """
    Returns the values of a forward pass of the network
    :param x: The input to feed into the network 
    """
    # 1. define first conv layer with max pooling
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)

    # 2. define second conv layer with max pooling
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)

    # 3. Define fully connected layers
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    x = self.relu3(x)
    x = self.fc2(x)
    return x



class ActorCritic(nn.Module):
    """The Actor Critic Neural Network used to estimate the state value function and action probabilities"""
    def __init__(self, s_size=8, h_size=128, a_size=4):
        
        # The network architecture follows the popular lenet-5 CNN architeture 
        super(ActorCritic, self).__init__()
        
        # Initialize first set of convolutional and pooling layers with a ReLU activation function 
        self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Initialize second set of of convolutional and pooling layers with a ReLU activation function 
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Initialize fully connected layers for glyph output after convolutional and pooling layers
        self.fc1 = Linear(in_features=1600, out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=128)
        self.relu4 = ReLU()
        
        # Initialize fully connected for message input 
        self.fc3 = Linear(in_features=256, out_features=128)
        self.relu5 = ReLU()
        
        # Initialize fully connected for combination of glyphs and message 
        self.fc4 = Linear(in_features=256, out_features=128)
        self.relu6 = ReLU()

        # To estimate the value function of the state 
        self.value_layer = nn.Linear(128, 1)

        # To calculate the probability of taking each action in the given state
        self.action_layer = nn.Linear(128, a_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        
        # Transform the glyph and state arrays into tensors 
        glyphs_t = torch.from_numpy(state["glyphs"]).float().to(device)
        message_t = torch.from_numpy(state["message"]).float().to(device)

        # Pass the 2D glyphs input through our convolutional and pooling layers 
        glyphs_t = self.conv1(glyphs_t)
        glyphs_t = self.relu1(glyphs_t)
        glyphs_t = self.maxpool1(glyphs_t)
        glyphs_t = self.conv2(glyphs_t)
        glyphs_t = self.relu2(glyphs_t)
        glyphs_t = self.maxpool2(glyphs_t)
        
        # Platten the output from the final pooling layer and pass it through the fully connected layers 
        glyphs_t = glyphs_t.reshape(glyphs_t.shape[0], -1)
        glyphs_t = self.fc1(glyphs_t)
        glyphs_t = self.relu3(glyphs_t)
        glyphs_t = self.fc2(glyphs_t)
        glyphs_t = self.relu4(glyphs_t)
        
        # Pass the message input through a fully connected layer
        message_t = self.fc3(message_t)
        message_t = self.relu5(message_t)
        
        # Combine glyphs output from convolution and fully connected layers 
        # with message output from fully connected layer 
        # Cat and Concat are used for different versions of PyTorch
        try:
            combined = torch.cat((glyphs_t, message_t), 1)
        except:
            combined = torch.concat([glyphs_t, message_t],1)

        # Pass glyphs and messaged combination through a fully connected layer
        combined = self.fc4(combined)
        combined = self.relu6(combined)
        
        # Pass the output from the previous fully connected layer through two seperate 
        # fully connected layers, one with a single output neuron (to estimate the state value function)
        # and the other with the number of output neurons equal to the number of actions 
        # (to estimate the action probabilities)
        state_value = self.value_layer(combined)
        
        action_probs = self.action_layer(combined)
        action_probs = self.softmax(action_probs)
        
        return action_probs, state_value
