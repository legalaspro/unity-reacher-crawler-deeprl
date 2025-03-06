import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


from unityagents import UnityEnvironment
from collections import deque



# Import SAC agent and training utilities
from rl_algo.sac_agent import Agent as SACAgent
from utils.crawler_train import train_agent

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC agent on Crawler environment')
    parser.add_argument('--total-steps', type=int, default=1000000,
                        help='Total environment steps to train for')
    parser.add_argument('--save-prefix', type=str, default='sac_crawler',
                        help='Prefix for saved model and data files')
    return parser.parse_args()

def plot_scores(scores, window=100):
    """Plot training progress."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot raw scores
    x = np.arange(len(scores))
    y = [score[0] for score in scores]  # Mean scores
    ax.plot(x, y, alpha=0.3, color='blue', label='Score')
    
    # Plot min and max ranges
    mins = [score[1] for score in scores]
    maxs = [score[2] for score in scores]
    ax.fill_between(x, mins, maxs, alpha=0.1, color='blue')
    
    # Plot moving average
    if len(scores) >= window:
        moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(len(moving_avg)) + window-1, moving_avg, color='blue', 
                label=f'{window}-episode Moving Avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    plt.savefig(f'{args.save_prefix}_training_curve.png')
    plt.close()
    return fig

def main(args):
    # Create Unity environment
    print(f"Loading environment from Crawler app")
    env = UnityEnvironment(file_name='./app/Crawler.app', worker_id=3, no_graphics=True)
    
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # Reset the environment to get state/action dimensions
    env_info = env.reset(train_mode=True)[brain_name]
    
    # Extract environment dimensions
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    print(f"Environment loaded: {num_agents} agents, State size: {state_size}, Action size: {action_size}")
    
    # Create SAC agent with optimized hyperparameters
    print("Creating SAC agent...")
    sacAgent = SACAgent(state_size=state_size, 
                    action_size=action_size, 
                    lr_actor=3e-4, 
                    lr_critic=3e-4,
                    skip_steps=1,
                    update_times=3,
                    gamma=0.99,
                    automatic_entropy_tuning=True,
                    learning_starts=20000)
    
    # Train the agent
    print(f"Training for {args.total_steps} steps...")
    rewards_hist, survival_hist = train_agent(
        env=env,
        agent=sacAgent,
        num_agents=num_agents,
        total_steps=args.total_steps,
        save_name=args.save_prefix
    )
    
    # Save training data
    print("Saving training data...")
    os.makedirs('data', exist_ok=True)
    np.save(f'data/{args.save_prefix}_rewards.npy', rewards_hist)
    np.save(f'data/{args.save_prefix}_survival.npy', survival_hist)
    
    # Plot results
    plot_scores(rewards_hist)
    
    # Close environment
    env.close()
    print("Training complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)