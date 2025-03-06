
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from rl_algo.ddpg_model import Actor, Critic
from rl_algo.replay_buffer import ReplayBuffer


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, 
                 random_seed=0, 
                 buffer_size=int(1e6),
                 batch_size=256,
                 gamma=0.99,
                 tau=0.001,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 exploration_noise=0.1, 
                 max_grad_norm=None,
                 skip_steps=1,
                 update_times=1,
                 learning_starts=10000):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size for learning
            gamma (float): discount factor
            tau (float): soft update parameter
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            exploration_noise (float): scale of exploration noise
            max_grad_norm (float): maximum gradient norm
            skip_steps (int): number of steps to skip before triggering learning
            update_times (int): number of update iterations per learning trigger
            learning_starts (int): minimum experiences before learning
        """
        self.state_size = state_size
        self.action_size = action_size
        self.exploration_noise = exploration_noise
        self.seed = np.random.seed(random_seed)
        
        # Hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.skip_steps = skip_steps
        self.update_times = update_times
        self.max_grad_norm = max_grad_norm
        self.learning_starts = learning_starts

        # Print all parameters
        self._print_parameters()

        # Track the time step
        self.t_step = 0
        self.agent_steps = 0
        self._training_mode = True

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed)
        self.actor_target = Actor(state_size, action_size, random_seed)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed)
        self.critic_target = Critic(state_size, action_size, random_seed)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)

        # Replay memory
        self.memory = ReplayBuffer(action_size, state_size, self.buffer_size, self.batch_size, random_seed)
    
    @property
    def training_mode(self):
        """Whether the agent is in training mode (vs inference mode)."""
        return self._training_mode

    def set_training(self, is_training):
        """Set the agent to training or inference mode."""
        self._training_mode = is_training
        if is_training:
            self.actor_local.train()
            self.critic_local.train()
        else:
            self.actor_local.eval()
            self.critic_local.eval()
        return self


    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        agents_num = state.shape[0]
        if agents_num > 1:
            self.memory.add_multiple(state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Increment the time step
        self.t_step += 1
        self.agent_steps += agents_num

        if self.t_step % self.skip_steps != 0:
            return

        # Learn, if enough samples are available in memory
        if  self.agent_steps > self.learning_starts and len(self.memory) > self.batch_size:
            for _ in range(self.update_times): #Update multiple times
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float()
        agents_num = state.shape[0]
        action_shape = (agents_num, self.action_size) if agents_num > 1 else (1, self.action_size)

        if self.training_mode and self.agent_steps < self.learning_starts:
            return np.random.uniform(-1., 1., action_shape)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise and self.training_mode:
            # Add noise to the action - Gaussian noise with mean 0 and standard deviation 0.1-0.2
            # its proven to be better then OU noise
            action += np.random.normal(0, self.exploration_noise, size=action_shape)

        action = np.clip(action, -1.0, 1.0)
        return action

    def reset(self):
        self.t_step = 0
        self.agent_steps = 0

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def save(self, filename):
        """Save the model parameters."""
        data = {
            "actor": self.actor_local.state_dict(),
            "critic": self.critic_local.state_dict(),
        }

        torch.save(data, filename)

    def load(self, filename):
        """Load the model parameters."""
        data = torch.load(filename)

        self.actor_local.load_state_dict(data["actor"])
        self.actor_target.load_state_dict(data["actor"])

        self.critic_local.load_state_dict(data["critic"])
        self.critic_target.load_state_dict(data["critic"])
    
    def _print_parameters(self):
        """Print all the hyperparameters for the DDPG agent."""
        separator = "-" * 50
        print(separator)
        print("DDPG Agent Parameters:")
        print(separator)
        print(f"State Size:         {self.state_size}")
        print(f"Action Size:        {self.action_size}")
        print(f"Random Seed:        {self.seed}")
        print(separator)
        print("Learning Parameters:")
        print(f"Buffer Size:        {self.buffer_size}")
        print(f"Batch Size:         {self.batch_size}")
        print(f"Gamma (Discount):   {self.gamma}")
        print(f"Tau (Soft Update):  {self.tau}")
        print(f"Actor LR:           {self.lr_actor}")
        print(f"Critic LR:          {self.lr_critic}")
        print(f"Exploration Noise:  {self.exploration_noise}")
        print(f"Max Grad Norm:      {self.max_grad_norm}")
        print(f"Learning Starts:    {self.learning_starts} steps")
        print(separator)


