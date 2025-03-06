import torch
import torch.nn as nn
import torch.nn.functional as F
import math


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)
         
        self._actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_size)
        )

        self.log2 = math.log(2)

        self.apply(weights_init_)
        # For final layer with tanh, use a small initialization:
        nn.init.xavier_uniform_(self._actor[4].weight, gain=1)
        nn.init.zeros_(self._actor[4].bias)


    def forward(self, x):
        mean, log_std = self._actor(x).chunk(2, dim=-1)
        # Constrain log_std inside [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # Compute std
        std = log_std.exp()
        return mean, std

    def get_action(self, obs, compute_log_pi=True, deterministic=False):
        mean, std = self.forward(obs)
        
        if deterministic:
            # used for evaluation of the policy 
            action = mean
            action = torch.tanh(action) 
            return action, None
        
        base_distribution = torch.distributions.Normal(mean, std)
        # additional transforms to run on top
        # First tanh to bound between [-1, 1]
        tanh_transform = torch.distributions.transforms.TanhTransform(cache_size=1)
        # Then scale and shift to match action space
        # scale_transform = torch.distributions.transforms.AffineTransform(self.action_bias, self.action_scale)
        squashed_dist = torch.distributions.TransformedDistribution(base_distribution, [tanh_transform,])

        # Get squashed action
        action = squashed_dist.rsample()

        if not compute_log_pi:
            return action, None
        
        log_prob = squashed_dist.log_prob(action).sum(-1, keepdim=True)
        # this equal to 
        # log_prob = base_distribution.log_prob(action)
        # log_prob -=  2*(np.log(2) - action - F.softplus(-2 * action))
        # log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


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
        x = torch.cat([x, a], dim=1)
        q1 = self._critic1(x)
        q2 = self._critic2(x)
        return q1, q2



