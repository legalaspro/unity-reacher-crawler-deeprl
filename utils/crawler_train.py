import numpy as np
from collections import deque

def detect_anomalies(states, actions, rewards, next_states):
    """
    Detect anomalies in states, actions, rewards, and next_states for multi-agent environments.
    Returns True if an anomaly is found, False otherwise.
    """
    # Check for NaN or infinity in all inputs
    for name, arr in [("states", states), ("actions", actions), ("rewards", rewards), ("next_states", next_states)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            # For multi-dimensional arrays, we need a different approach
            is_invalid = np.isnan(arr) | np.isinf(arr)
            print(f"WARNING: Invalid {name} detected")
            
            # For rewards specifically, print more details (usually 1D per agent)
            if name == "rewards":
                invalid_indices = np.where(is_invalid)[0]
                if len(invalid_indices) > 0:
                    print(f"  Invalid reward indices: {invalid_indices}")
                    print(f"  Invalid reward values: {arr[invalid_indices]}")
            return True

    # Check for extreme rewards (finite but potentially destabilizing)
    reward_threshold = 1000.0
    extreme_indices = np.where(np.abs(rewards) > reward_threshold)[0]
    if len(extreme_indices) > 0:
        print(f"WARNING: Extreme rewards detected at indices {extreme_indices}: {rewards[extreme_indices]}")
        return True
    
    # Check for extreme state values
    state_threshold = 100.0
    if np.any(np.abs(states) > state_threshold):
        print(f"WARNING: Extreme state values detected")
        return True

    return False

def train_agent(env, agent, num_agents=12, max_t=1000, total_steps=1000000, log_every=1, save_name=None, check_anomalies=False):
    """
    Train a Agent for the Unity Crawler environment with 12 agents under one brain.
    
    Args:
        env: Unity environment
        agent: RL agent (DDPG, TD3, etc.)
        num_agents: Number of agents (12 for Crawler)
        max_t: Maximum timesteps per episode
        total_steps: Total number of steps to train
        log_every: How often to print logs
        save_name: Base name for saving models
    
    Returns:
        rewards_hist: List of [mean, min, max] rewards per episode
        survival_hist: List of [mean, min, max] survival times per episode
    """
    brain_name = env.brain_names[0]
    rewards_hist = []
    survival_hist = []
    scores_window = deque(maxlen=100)
    length_window = deque(maxlen=1000)
    best_score = -np.inf
    anomaly_count = 0

    # Track total steps and episodes
    total_step_count = 0
    episode = 0

    while total_step_count < total_steps:
        episode += 1
        total_reward = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        
        agent_steps = np.zeros(num_agents)
        episode_survivals = []
        done_mask = np.ones(num_agents, dtype=bool)  # Track active agents

        for i in range(max_t):
            # Add per-agent exploration noise
            actions = agent.act(states)
            actions = np.where(done_mask[:, None], actions, 0)  # Zero actions for done agents

            env_info = env.step(actions)[brain_name]
            agent_steps += done_mask  # Only increment steps for active agents
            total_step_count += done_mask.sum()

            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards)
            dones = np.array(env_info.local_done)

            # Check anomalies before clipping
            if check_anomalies and detect_anomalies(states, actions, rewards, next_states):
                anomaly_count += 1
                print(f"Episode {episode}, Step {i}: Anomaly #{anomaly_count}")
                states = next_states
                continue

            # Only use active agents for learning
            active_indices = np.where(done_mask)[0]
            if len(active_indices) > 0:  # If we have any active agents
                agent.step(
                    states[active_indices],
                    actions[active_indices], 
                    rewards[active_indices], 
                    next_states[active_indices], 
                    dones[active_indices]
                )

            total_reward += rewards * done_mask  # Only count rewards for active agents
            states = next_states

            # Handle partial terminations
            if np.any(dones):
                done_indices = np.where(dones & done_mask)[0]
                for idx in done_indices:
                    episode_survivals.append(agent_steps[idx])
                    length_window.append(agent_steps[idx])
                    agent_steps[idx] = 0
                    done_mask[idx] = False  # Mark agent as inactive

            # Early reset if all agents fail
            if np.all(~done_mask):
                # print(f"Episode {episode:4d} ended at step {i:4d} - All agents done")
                break

        # Record survival for agents still active
        for idx in range(num_agents):
            if done_mask[idx]:
                episode_survivals.append(agent_steps[idx])
                length_window.append(agent_steps[idx])

        mean_reward = np.mean(total_reward)
        min_reward = np.min(total_reward)
        max_reward = np.max(total_reward)
        rewards_hist.append([mean_reward, min_reward, max_reward])
        scores_window.append(mean_reward)

        mean_survival = np.mean(episode_survivals) if episode_survivals else max_t
        min_survival = np.min(episode_survivals or [max_t])
        max_survival = np.max(episode_survivals or [max_t])
        survival_hist.append([mean_survival, min_survival, max_survival])

        if episode % log_every == 0:
            avg_survival = np.mean(length_window) if length_window else 0
            print(f'Episode {episode:3d}({total_step_count:7d}) | '
                  f'Avg-100: {np.mean(scores_window):.2f} | '
                  f'Score: {mean_reward:.2f} ({min_reward:.2f} ; {max_reward:.2f}) | '
                  f'Length: {mean_survival:.1f} ({min_survival} ; {max_survival}) | '
                  f'Avg-1000L: {avg_survival:.1f}')

        # Save best model
        if np.mean(scores_window) > best_score and save_name:
            best_score = np.mean(scores_window)
            agent.save(f"{save_name}_best.pth")
            # print(f"\n*** New best score: {best_score:.3f} (model saved) ***\n")

    return rewards_hist, survival_hist