import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_size)

       
        self.apply(weights_init_)
        # For final layer with tanh, use a small initialization:
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=1)
        nn.init.zeros_(self.fc_mu.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()

        inputs_dim = state_size + action_size
        self.seed = torch.manual_seed(seed)
        
        self._critic1 = nn.Sequential(
            nn.Linear(inputs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._critic2 = nn.Sequential(
            nn.Linear(inputs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.apply(weights_init_)

    def forward(self, x, a):
        assert x.shape[0] == a.shape[0]
        assert len(x.shape) == 2 and len(a.shape) == 2
        x = torch.cat([x, a], 1)
        q1 = self._critic1(x)
        q2 = self._critic2(x)
        return q1, q2

