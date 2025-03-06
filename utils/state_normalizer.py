import numpy as np

class StateNormalizer:
    """Normalize state inputs using running statistics."""
    
    def __init__(self, state_size, epsilon=1e-4, clip_range=5.0):
        """Initialize state normalizer.
        
        Args:
            state_size: Dimension of the state space
            epsilon: Small constant to avoid division by zero
            clip_range: Normalized values are clipped to ±clip_range
        """
        self.state_size = state_size
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # Running count
        self.total_count = 0
        
        # Running sum and squared sum for computing mean and variance
        self.sum = np.zeros(state_size, dtype=np.float64)
        self.sum_sq = np.zeros(state_size, dtype=np.float64)
        
        # Mean and std (initialized to identity normalization)
        self.mean = np.zeros(state_size, dtype=np.float64)
        self.std = np.ones(state_size, dtype=np.float64)
        
    def update(self, states):
        """Update running statistics with new state observations."""
        if isinstance(states, list):
            states = np.array(states)
        
        # Handle both single states and batches
        if states.ndim == 1:
            states = states.reshape(1, -1)
            
        batch_count = states.shape[0]
        batch_sum = states.sum(axis=0)
        batch_sum_sq = (states ** 2).sum(axis=0)
        
        # Update running stats
        self.sum += batch_sum
        self.sum_sq += batch_sum_sq
        self.total_count += batch_count
        
        # Recalculate mean and standard deviation
        self.mean = self.sum / self.total_count
        self.std = np.sqrt(
            np.maximum(
                self.epsilon,  # Avoid division by zero
                self.sum_sq / self.total_count - (self.mean ** 2)
            )
        )
    
    def normalize(self, states):
        """Normalize states using current statistics."""
        normalized = (states - self.mean) / self.std
        
        # Clip to reasonable range to prevent outliers
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def save(self, path):
        """Save normalizer parameters to file."""
        np.savez(path, 
                 mean=self.mean, 
                 std=self.std,
                 sum=self.sum,
                 sum_sq=self.sum_sq,
                 count=np.array([self.total_count]))
        
    def load(self, path):
        """Load normalizer parameters from file."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.sum = data['sum']
        self.sum_sq = data['sum_sq']
        self.total_count = int(data['count'][0])