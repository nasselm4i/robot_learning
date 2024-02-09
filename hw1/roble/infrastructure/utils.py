import numpy as np
import time
import os 
############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # DONE : initialize env for the beginning of a new rollout
    #  HINT: should be the output of resetting the env
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        # use the most recent ob to decide what to do
        obs.append(ob)
        # DONE: query the policy's get_action function
        # HINT: query the policy's get_action function
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)
        # print("Dim of action: ", ac.shape)
        # print("Action: ", ac)
        ob, rew, done, _ = env.step(ac)
        
        # record result of taking that action
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1

        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0

        # DONE end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        DONE implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch <= min_timesteps_per_batch:
        sample_path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(sample_path)
        timesteps_this_batch += get_pathlength(sample_path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        DONE implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []
    for _ in range(ntraj):
        sample_path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(sample_path)
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def flatten(matrix):
    ## Check and fix a matrix with different length lists.
    import collections.abc
    if (isinstance(matrix, (collections.abc.Sequence,))  and 
        isinstance(matrix[0], (collections.abc.Sequence, np.ndarray))): ## Flatten possible inhomogeneous arrays
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list
    else:
        return matrix
    
import matplotlib.pyplot as plt

def make_figure(data, title, window_size=50):
    values = [list(d.values())[0] for d in data]
    # smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    plt.figure()
    plt.plot(values) 
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{title}.png")
    print(f"Saved {title}.png in the directory: {os.getcwd()}")
    plt.close()
    return f"{title}.png"

# a learning curve with DAgger (Dataset Aggregation) iterations versus the policy's mean return, including error bars for the standard deviation
# The x-axis will represent the number of DAgger iterations.
# The y-axis will show the policy's mean return.
# Error bars will indicate the standard deviation at each point, showing the variation in returns.
def plot_dagger_learning_curve(dagger_iterations, mean_returns, std_dev, title):
    # Flatten mean_returns if it's a nested list or a multidimensional array
    # if isinstance(mean_returns, list):
    #     mean_returns = [val for sublist in mean_returns for val in sublist]
    if isinstance(mean_returns, np.ndarray) and mean_returns.ndim > 1:
        mean_returns = mean_returns.flatten()

    # Ensure all arrays have the same length
    assert len(dagger_iterations) == len(mean_returns) == len(std_dev), "All input arrays must have the same length"

    # Plotting code
    plt.figure(figsize=(10, 6))
    plt.errorbar(dagger_iterations, mean_returns, yerr=std_dev, fmt='-o', capsize=5)
    plt.xlabel('DAgger Iterations')
    plt.ylabel('Policy Mean Return')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{title}.png")
    plt.show()