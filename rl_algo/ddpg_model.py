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
        self.fc1 = nn.Linear(state_size, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc_mu = nn.Linear(300, action_size)

        self.apply(weights_init_)
        # For final layer with tanh, use a small initialization:
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=1)
        nn.init.zeros_(self.fc_mu.bias)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc_mu(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(action_size + 400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)

       
        self.apply(weights_init_)

    def forward(self, x, a):
        assert x.shape[0] == a.shape[0]
        assert len(x.shape) == 2 and len(a.shape) == 2
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(torch.cat([x, a], 1))))
        x = self.fc3(x)
        return x


