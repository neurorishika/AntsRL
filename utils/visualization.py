import matplotlib.pyplot as plt
import numpy as np

def plot_grid_and_odor(env_state, ax=None):
    """
    Visualizes the grid, zones, agent position, and odor intensity.

    Args:
        env_state (dict): Dictionary containing env info like grid_odor,
                          agent_pos, home_zone, goal_zone etc.
                          Usually obtained from env.render() or env.get_info().
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    N = env_state['grid_odor'].shape[0]
    grid_odor = env_state['grid_odor']
    agent_pos = env_state['agent_pos']
    agent_idx = env_state['current_agent_idx']
    home_zone = env_state['home_zone']
    goal_zone = env_state['goal_zone']

    # Plot odor intensity as background
    im = ax.imshow(grid_odor, cmap='Blues', origin='lower', interpolation='nearest', vmin=0)
    plt.colorbar(im, ax=ax, label='Odor Intensity')

    # Plot zones
    for r, c in home_zone:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.7, label='Home Zone' if (r,c) == home_zone[0] else "")
        ax.add_patch(rect)
    for r, c in goal_zone:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='b', facecolor='none', alpha=0.7, label='Goal Zone' if (r,c) == goal_zone[0] else "")
        ax.add_patch(rect)


    # Plot agent
    agent_color = 'red' if agent_idx == 0 else 'orange'
    ax.plot(agent_pos[1], agent_pos[0], 'o', markersize=10, color=agent_color, label=f'Agent {agent_idx+1}')

    # Plot agent heading (simple arrow)
    heading = env_state['agent_heading']
    dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][heading] # N: (0,1), E:(1,0), S:(0,-1), W:(-1,0) - Mapping needs adjustment for imshow origin='lower'
    dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][heading] # N: (0,1) E:(1,0) S:(0,-1) W:(-1,0)
    # Correcting for imshow y-axis inversion: N=(0,1) -> dy=1 is Up; S=(0,-1) -> dy=-1 is Down. Correct.
    # Correcting for imshow x-axis: E=(1,0) -> dx=1 is Right; W=(-1,0) -> dx=-1 is Left. Correct.
    ax.arrow(agent_pos[1], agent_pos[0], dx*0.4, dy*0.4, head_width=0.3, head_length=0.3, fc=agent_color, ec=agent_color)


    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(0, N, max(1, N//5))) # Major ticks
    ax.set_yticks(np.arange(0, N, max(1, N//5)))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_title(f"Grid State (Agent {agent_idx+1}'s turn)")
    ax.legend(loc='upper right')

    # Draw title/info?
    # ax.set_title(f"Step: {env_state['total_steps']}, Agent: {agent_idx+1}")

    if ax is None:
        plt.show()


def plot_episode_trajectory(trajectory_info, grid_size, home_zone, goal_zone, title="Episode Trajectory"):
    """
    Plots the full trajectory of both agents for an episode.

    Args:
        trajectory_info (list): List of info dicts collected during an episode.
        grid_size (int): N.
        home_zone (list): Coordinates.
        goal_zone (list): Coordinates.
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot zones first
    for r, c in home_zone:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.3, label='Home Zone' if (r,c) == home_zone[0] else "")
        ax.add_patch(rect)
    for r, c in goal_zone:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='b', facecolor='none', alpha=0.3, label='Goal Zone' if (r,c) == goal_zone[0] else "")
        ax.add_patch(rect)

    # Extract trajectories
    agent1_pos = []
    agent2_pos = []
    current_agent = 0
    for info in trajectory_info:
        if info['current_agent_idx'] == 0:
            agent1_pos.append(info['agent_pos'][::-1]) # Flip to (x, y) for plotting
        else:
            # If this is the first step for agent 2, add agent 1's last known pos
            if current_agent == 0 and agent1_pos:
                 agent1_pos.append(info['agent_pos'][::-1]) # Agent 1 'ends' where Agent 2 starts

            agent2_pos.append(info['agent_pos'][::-1])
        current_agent = info['current_agent_idx']


    if agent1_pos:
        agent1_pos = np.array(agent1_pos)
        ax.plot(agent1_pos[:, 0], agent1_pos[:, 1], 'r-', label='Agent 1 Traj', alpha=0.8)
        ax.plot(agent1_pos[0, 0], agent1_pos[0, 1], 'ro', markersize=8, label='A1 Start')
        if trajectory_info[-1].get('agent1_returned_home', False):
             ax.plot(agent1_pos[-1, 0], agent1_pos[-1, 1], 'rX', markersize=10, label='A1 End (Returned)')


    if agent2_pos:
        agent2_pos = np.array(agent2_pos)
        ax.plot(agent2_pos[:, 0], agent2_pos[:, 1], 'o-', color='orange', label='Agent 2 Traj', alpha=0.8, markersize=4)
        ax.plot(agent2_pos[0, 0], agent2_pos[0, 1], 'o', color='orange', markersize=8, label='A2 Start')
        if trajectory_info[-1].get('agent2_reached_goal', False):
             ax.plot(agent2_pos[-1, 0], agent2_pos[-1, 1], 'X', color='orange', markersize=10, label='A2 End (Goal)')


    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(0, grid_size, max(1, grid_size//5))) # Major ticks
    ax.set_yticks(np.arange(0, grid_size, max(1, grid_size//5)))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_title(title)
    ax.legend(loc='best')
    plt.show()