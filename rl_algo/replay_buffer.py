
import torch
import random
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size,buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        
        # Initialize buffer with zeros
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32) 
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)  
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        # Position management
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Store experience in buffer
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update position and size
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
     
    def add_multiple(self, states, actions, rewards, next_states, dones):
        """Add multiple experiences to memory efficiently."""
        batch_size = len(states)
        
        # Handle case where adding experiences would exceed buffer size
        if self.position + batch_size <= self.buffer_size:
            # Simple case: continuous segment
            self.states[self.position:self.position+batch_size] = states
            self.actions[self.position:self.position+batch_size] = actions
            self.rewards[self.position:self.position+batch_size] = np.expand_dims(rewards, 1)
            self.next_states[self.position:self.position+batch_size] = next_states
            self.dones[self.position:self.position+batch_size] = np.expand_dims(dones, 1)
            
            # Update position and size
            self.position = (self.position + batch_size) % self.buffer_size
            self.size = min(self.size + batch_size, self.buffer_size)
        else:
            # Handle wrap-around case: split into two segments
            first_segment = self.buffer_size - self.position
            second_segment = batch_size - first_segment
            
            # First segment
            self.states[self.position:] = states[:first_segment]
            self.actions[self.position:] = actions[:first_segment]
            self.rewards[self.position:] = np.expand_dims(rewards[:first_segment], 1)
            self.next_states[self.position:] = next_states[:first_segment]
            self.dones[self.position:] = np.expand_dims(dones[:first_segment], 1)
            
            # Second segment (wrapping to beginning)
            self.states[:second_segment] = states[first_segment:]
            self.actions[:second_segment] = actions[first_segment:]
            self.rewards[:second_segment] = np.expand_dims(rewards[first_segment:], 1)
            self.next_states[:second_segment] = next_states[first_segment:]
            self.dones[:second_segment] = np.expand_dims(dones[first_segment:], 1)
            
            # Update position and size
            self.position = second_segment
            self.size = self.buffer_size
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # Generate random indices
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        
        # Get batch directly from numpy arrays
        states = torch.from_numpy(self.states[indices]).float()
        actions = torch.from_numpy(self.actions[indices]).float()
        rewards = torch.from_numpy(self.rewards[indices]).float()
        next_states = torch.from_numpy(self.next_states[indices]).float()
        dones = torch.from_numpy(self.dones[indices]).float()
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size