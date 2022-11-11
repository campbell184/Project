import math
import torch
import random
import numpy as np
import torch.nn as nn
from gym import spaces
from model import DQN
import torch.nn.functional as F
from replay_buffer import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class that brings all the DQN compenents together so that a model can be trained
class DQNAgent():
  def __init__(self, observation_space, action_space, **kwargs):
    global device
    self.action_space = action_space
    self.replay_buffer = kwargs.get("replay_buffer", None)
    self.use_double_dqn = kwargs.get("use_double_dqn", None)
    self.gamma = kwargs.get("gamma", 0.99)
    self.lr = kwargs.get("lr", None)
    self.betas = kwargs.get("betas", (0.9, 0.999))
    self.batch_size = kwargs.get("batch_size", None)

    # Create the policy and target network
    self.policy_network = DQN(action_space).to(device)
    self.target_network = DQN(action_space).to(device)
    self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr, betas=self.betas)
   
  def optimise_td_loss(self):
    """
    Optimise the TD-error over a single minibatch of transitions
    :return: the loss
    """
    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
    states = np.array(states)
    next_states = np.array(next_states)
    states = torch.from_numpy(states).float().to(device)
    actions = torch.from_numpy(actions).long().to(device)
    rewards = torch.from_numpy(rewards).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    dones = torch.from_numpy(dones).float().to(device)

    with torch.no_grad():
      if self.use_double_dqn:
        _, max_next_action = self.policy_network(next_states).max(1)
        max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
      else:
        next_q_values = self.policy_network(next_states)
        max_next_q_values, _ = next_q_values.max(1)
      target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

    input_q_values = self.target_network(states)
    input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()
    loss = F.smooth_l1_loss(input_q_values, target_q_values)

    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    del states
    del next_states
    return loss.item()

  def update_target_network(self):
    """
    Update the target Q-network by copying the weights from the current Q-network
    """
    self.target_network.load_state_dict(self.policy_network.state_dict())

  def act(self, observation):
    """Select action base on network inference"""
    if not torch.cuda.is_available():
        observation = observation.type(torch.FloatTensor) 
    else:
        observation = observation.type(torch.cuda.FloatTensor) 
    state = torch.unsqueeze(observation, 0).to(device)
    result = self.policy_network.forward(state)
    action = torch.argmax(result).item()
    return action