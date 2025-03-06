import numpy as np
import matplotlib.pyplot as plt

def plot_scores(rewards_history, window_size=10, target_score=30.0, figsize=(12, 8), title=None):
    """
    Plot training scores over time with averages and target line.
    
    Args:
        rewards_history: List of [mean, min, max] for each episode
        window_size: Size of window for running average
        target_score: Score threshold line to display
        figsize: Figure dimensions
    
    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    episodes = np.arange(1, len(rewards_history) + 1)
    means = [r[0] for r in rewards_history]
    mins = [r[1] for r in rewards_history]
    maxs = [r[2] for r in rewards_history]
    
    # Calculate running average
    avg_means = []
    
    for i in range(len(means)):
        if i < window_size:
            avg_means.append(np.mean(means[:i+1]))
        else:
            avg_means.append(np.mean(means[i-window_size+1:i+1]))
    
    # Plot the min-max range
    ax.fill_between(episodes, mins, maxs, alpha=0.2, color='blue', label='Min-Max Range')
    
    # Plot the mean scores
    ax.plot(episodes, means, 'o-', markersize=2, alpha=0.7, color='blue', label='Episode Score')
    
    # Plot the running average
    ax.plot(episodes, avg_means, linewidth=3, color='red', label=f'{window_size}-Episode Average')
    
    # Plot target line
    ax.axhline(y=target_score, color='green', linestyle='--', linewidth=2, label=f'Target Score ({target_score})')
    
    # Customize the plot
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    
    plot_title = title if title is not None else 'Training Scores Over Time'
    ax.set_title(plot_title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


def plot_episode_lengths(lengths_history, window_size=100, target_length=None, figsize=(12, 8), title=None):
    """
    Plot episode lengths over time with min, max, and mean values.
    
    Args:
        lengths_history: List of [mean, min, max] for each episode
        window_size: Size of window for running average
        target_length: Optional target length to display as horizontal line
        figsize: Figure dimensions
    
    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Default fallback
    fill_color = '#A8D0E6'  # Light blue
    line_color = '#2E86C1'  # Medium blue
    avg_color = '#1A5276'   # Dark blue
    
    # Extract data
    episodes = np.arange(1, len(lengths_history) + 1)
    means = [r[0] for r in lengths_history]
    mins = [r[1] for r in lengths_history]
    maxs = [r[2] for r in lengths_history]
    
    # Calculate running average
    avg_means = []
    for i in range(len(means)):
        if i < window_size:
            avg_means.append(np.mean(means[:i+1]))
        else:
            avg_means.append(np.mean(means[i-window_size+1:i+1]))
    
    # Plot the min-max range
    ax.fill_between(episodes, mins, maxs, alpha=0.2, color=fill_color, label='Min-Max Range')
    
    # Plot the mean episode lengths
    ax.plot(episodes, means, 'o-', markersize=2, alpha=0.7, color=line_color, label='Episode Length')
    
    # Plot the running average
    ax.plot(episodes, avg_means, linewidth=3, color=avg_color, label=f'{window_size}-Episode Average')
    
    # Plot optional target line
    if target_length is not None:
        ax.axhline(y=target_length, color='red', linestyle='--', linewidth=2, 
                  label=f'Target Length ({target_length})')
    
    # Customize the plot
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)

    plot_title = title if title is not None else 'Episode Lengths Over Time'
    ax.set_title(plot_title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_comparison(file_paths, labels, colors=None, window_size=100, target_score=30, figsize=(14, 8)):
    """
    Plot multiple training runs together for comparison, handling multi-agent data.
    
    Args:
        file_paths: List of .npy files containing rewards history
        labels: List of labels for each run
        colors: Optional list of colors for each run
        window_size: Window size for running average (in terms of episodes)
        target_score: Target score to display as horizontal line
        figsize: Figure dimensions
    
    Returns:
        fig: The matplotlib figure object
        all_means: Dictionary of raw means for each run
        all_avg_means: Dictionary of smoothed averages for each run
    """
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    all_means = {}
    all_avg_means = {}
    
    # Load and plot each run
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        rewards_history = np.load(file_path, allow_pickle=True)
        color = colors[i % len(colors)]
        
        # Extract mean rewards (first column of each entry)
        means = np.array([r[0] for r in rewards_history])
        num_episodes = len(means)
        episodes = np.arange(1, num_episodes + 1) 

        # Store raw means
        all_means[label] = means
        
        # Calculate moving average over window_size episodes
        avg_means = np.zeros_like(means, dtype=float)
        
        # For each point, calculate average of previous window_size episodes
        for j in range(len(means)):
            start_idx = max(0, j - window_size + 1)
            window = means[start_idx:j + 1]
            avg_means[j] = np.mean(window)
        
        all_avg_means[label] = avg_means
        
        # Plot smoothed averages (main line)
        ax.plot(episodes, avg_means, linewidth=2.5, color=color, label=f'{label} ({window_size}-ep avg)')
        
        # Plot raw means with lower opacity
        ax.plot(episodes, means, linewidth=1.0, alpha=0.5, color=color, 
                label=f'{label} (mean)', linestyle='--')

    # Add target score line
    ax.axhline(y=target_score, color='black', linestyle='--', linewidth=2,
               label=f'Target Score ({target_score})')
    
    # Customize the plot
    ax.set_xlabel('Agent Episodes', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(f'Algorithm Comparison ({window_size}-episode moving average)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # Adjust x-axis limits
    max_episodes = max([len(np.load(fp, allow_pickle=True)) for fp in file_paths])
    plt.xlim([0, (max_episodes + 5)])
    
    plt.tight_layout()
    return fig, all_means, all_avg_means