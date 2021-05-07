import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """ Initialize an Agent object.
        
            INPUTS: 
            ------------
                state_size - (int) dimension of each state
                action_size - (int) dimension of each action
                seed - (int) random seed
            
            OUTPUTS:
            ------------
                no direct
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        
            INPUTS: 
            ------------
                state - (array_like) the previous state of the environment (8,)
                action - (int) the agent's previous choice of action
                reward - (float) last reward received
                next_state - (torch tensor) the current state of the environment
                done - (bool) whether the episode is complete (True or False)
            
            OUTPUTS:
            ------------
                no direct
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                #print('experiences')
                #print(experiences)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """ Returns actions for given state as per current policy.
        
            INPUTS:
            ------------
                state - (numpy array_like) current state
                eps - (float) epsilon, for epsilon-greedy action selection

            OUTPUTS:
            ------------
                act_select - (int) next epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
      
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            act_select = np.argmax(action_values.cpu().data.numpy())
            return act_select
        else:
            act_select = random.choice(np.arange(self.action_size))
            return act_select

    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples.

            INPUTS:
            ------------
                experiences - (Tuple[torch.Variable]) tuple of (s, a, r, s', done) tuples 
                gamma - (float) discount factor

            OUTPUTS:
            ------------
        """
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #print(Q_targets_next)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #print(Q_targets)
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #print(Q_expected)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            INPUTS:
            ------------
                local_model - (PyTorch model) weights will be copied from
                target_model - (PyTorch model) weights will be copied to
                tau - (float) interpolation parameter 
                
            OUTPUTS:
            ------------
                no direct
                
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Initialize a ReplayBuffer object.

        INPUTS:
        ------------
            action_size - (int) dimension of each action
            buffer_size - (int) maximum size of buffer
            batch_size - (int) size of each training batch
            seed - (int) random seed
            
        OUTPUTS:
        ------------
            no direct
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory.
            
            INPUTS:
            ------------
                state - (array_like) the previous state of the environment
                action - (int) the agent's previous choice of action
                reward - (int) last reward received
                next_state - (int) the current state of the environment
                done - (bool) whether the episode is complete (True or False)

            OUTPUTS:
            ------------
                no direct
        
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """ Randomly sample a batch of experiences from memory.
        
            INPUTS:
            ------------
                None
            
            OUTPUTS:
            ------------
                states - (torch tensor) the previous states of the environment
                actions - (torch tensor) the agent's previous choice of actions
                rewards - (torch tensor) last rewards received
                next_states - (torch tensor) the next states of the environment
                dones - (torch tensor) bools, whether the episode is complete (True or False)
        
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Return the current size of internal memory.
        
            INPUTS:
            ------------
                None
                
            OUTPUTS:
            ------------
                mem_size - (int) current size of internal memory
        
        """
        mem_size = len(self.memory)
        return mem_size