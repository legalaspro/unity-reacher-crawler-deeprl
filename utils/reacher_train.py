
import numpy as np
from collections import deque

def train_agent(env, agent, num_agents, max_t=1000, episodes=150, log_every=1, 
                state_normalizer=None, save_name=None):
    """Train an agent in the environment with detailed logging.
    
    Args:
        env: Unity environment
        agent: RL agent (DDPG, TD3, etc.)
        num_agents: Number of agents in the environment
        max_t: Maximum timesteps per episode
        episodes: Number of episodes to train
        log_every: How often to print logs
        state_normalizer: Optional state normalizer
    
    Returns:
        rewards_hist: List of [mean, min, max] rewards for each episode
    """
    def normalize_state(state):
        if state_normalizer is not None:
            state_normalizer.update(state)
            return state_normalizer.normalize(state)
        else:
            return state
    
    brain_name = env.brain_names[0]
    rewards_hist = []
    scores_window = deque(maxlen=100)
    best_score = -np.inf
    
    for episode in range(1, episodes+1): 
        total_reward = np.zeros(num_agents) # Reset the total reward

        # Reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = normalize_state(env_info.vector_observations)

        for i in range(max_t):
            # Select and execute actions
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            
            # Get new state and reward information
            next_states = normalize_state(env_info.vector_observations)
            rewards = env_info.rewards
            dones = env_info.local_done

            # Agent learning step
            agent.step(states, actions, rewards, next_states, dones)
            
            # Update total reward and roll over states
            total_reward += rewards
            states = next_states

            # seems like we have autorest in env.step
            # if np.any(dones):
            #     print(f"Step {i}, Agents done this step: {np.where(dones)[0]}")
            #     print(f"Step {i}, Total done agents so far: {np.where(~active_agents)[0]}")
            
            if np.all(dones):
                print(f'Episode {episode:4d} ended at step {i:4d} - All agents simultaneously done')
                break
        
        # Process episode result
        if num_agents > 1:
            # Multi-agent case - show statistics across agents
            mean_reward = np.mean(total_reward)
            min_reward = np.min(total_reward)
            max_reward = np.max(total_reward)
            rewards_hist.append([mean_reward, min_reward, max_reward])
            scores_window.append(mean_reward)
            
            if episode % log_every == 0:
                print(f'Episode {episode:4d} | '
                    f'Avg-100: {np.mean(scores_window):.3f} | '
                    f'Score: {mean_reward:.3f} | Min: {min_reward:.3f} | '
                    f'Max: {max_reward:.3f}')
        else:
            # Single agent case - simpler logging
            reward = total_reward[0]  # Extract the single value
            rewards_hist.append([reward, reward, reward])  # Keep same format for consistency
            scores_window.append(reward)
            
            if episode % log_every == 0:
                print(f'Episode {episode:4d} | '
                    f'Avg-100: {np.mean(scores_window):.3f} | '
                    f'Score: {reward:.3f}')

        # Save best model
        if np.mean(scores_window) > best_score and save_name:
            best_score = np.mean(scores_window)
            agent.save(f"{save_name}_best.pth")
            # print(f"\n*** New best score: {best_score:.3f} (model saved) ***\n")
            
        # Check if environment is solved
        # if np.mean(scores_window) >= target_score:
        #     print(f'\nEnvironment solved in {episode:d} episodes! '
        #           f'Average score: {np.mean(scores_window):.3f}')
        #     if save_path:
        #         torch.save({
        #             'actor': agent.actor_local.state_dict(),
        #             'critic': agent.critic_local.state_dict()
        #         }, f"{save_name}_solved.pth")
        #     break

    return rewards_hist




