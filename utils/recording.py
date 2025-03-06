import imageio
import time
import os
import numpy as np
import PIL.ImageGrab as ImageGrab
# from moviepy.editor import VideoFileClip


def play_single_episode(agent, env, brain_name, max_steps=1000):
    """Play a single episode with the agent without adding exploration noise.
    
    Args:
        agent: The trained agent with act() method
        env: Unity environment
        brain_name: Name of the Unity brain
        max_steps: Maximum number of steps per episode
        
    Returns:
        scores: Final score for each agent
    """
    # Set agent to evaluation mode
    agent.set_training(False)
    
    # Reset environment
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(len(env_info.agents))
    
    # Run episode
    for t in range(max_steps):
        # Select actions and step environment
        actions = agent.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]
        
        # Update states and scores
        scores += env_info.rewards
        states = env_info.vector_observations
        
        # Break if done
        if np.any(env_info.local_done):
            break
            
    print(f'Total score (averaged over agents): {np.mean(scores):.2f}')
    try:
        env.close()
        print("Environment closed successfully")
    except:
        print("Warning: Could not close environment")
    return scores

def convert_to_mp4(input_file, output_file=None, remove_original=False):
    """Convert a video file to MP4 format using imageio."""
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.mp4'
    
    print(f"Converting {input_file} to MP4...")
    try:
        reader = imageio.get_reader(input_file)
        fps = reader.get_meta_data()['fps']
        
        writer = imageio.get_writer(output_file, fps=fps)
        for frame in reader:
            writer.append_data(frame)
        writer.close()
        reader.close()
        
        print(f"Successfully converted to {output_file}")
        if remove_original:
            os.remove(input_file)
        
        return output_file
    except Exception as e:
        print(f"Error converting to MP4: {e}")
        return None

def convert_to_gif_simple(input_file, output_file=None, fps=10):
    """Convert a video to GIF without resizing - simple version."""
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.gif'
    
    print(f"Converting {input_file} to GIF at {fps} fps...")
    try:
        # Read video and write directly to GIF
        reader = imageio.get_reader(input_file)
        writer = imageio.get_writer(output_file, fps=fps)
        
        # Skip frames to reduce size
        num_frames = reader.count_frames()
        every_nth = max(1, num_frames // 100)  # Target ~100 frames in the GIF
        
        for i, frame in enumerate(reader):
            if i % every_nth == 0:
                writer.append_data(frame)
                
        writer.close()
        print(f"Successfully converted to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error converting to GIF: {e}")
        return None
    

def convert_to_gif(input_file, output_file=None, fps=10, skip_frames=3, scale_factor=0.5):
    """
    Convert video to GIF using only imageio.
    
    Args:
        input_file: Path to input video
        output_file: Path to output GIF (default: same name with .gif extension)
        fps: Frames per second in the output GIF
        skip_frames: Only use every Nth frame (reduces file size)
        scale_factor: Scale down factor (0.5 = half size)
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.gif'
    
    print(f"Converting {input_file} to GIF...")
    try:
        # Read the video
        reader = imageio.get_reader(input_file)
        
        # Collect frames with size reduction
        frames = []
        for i, frame in enumerate(reader):
            # Skip frames to reduce size
            if i % skip_frames != 0:
                continue
                
            # Resize using numpy slicing (simple but effective)
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Simple resize by taking every Nth pixel (fast but lower quality)
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            resized_frame = frame[::step_h, ::step_w]
            
            frames.append(resized_frame)
            
            # Progress indicator
            if i % 50 == 0:
                print(f"Processed frame {i}")
        
        print(f"Writing GIF with {len(frames)} frames...")
        # Save as GIF
        imageio.mimsave(output_file, frames, fps=fps)
        print(f"Successfully saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"Error converting to GIF: {e}")
        import traceback
        traceback.print_exc()
        return None