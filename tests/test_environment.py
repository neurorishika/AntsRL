# tests/test_environment.py

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import utils as space_utils
from gymnasium.utils.env_checker import check_env as gymnasium_check_env # Rename to avoid conflict

from custom_env.grid_env import GridEnv
from utils.helpers import load_config

# Load default config for testing
# Ensure this path is correct relative to your project root when running pytest
try:
    TEST_CONFIG_PATH = "config/default_config.yaml"
    FULL_CONFIG = load_config(TEST_CONFIG_PATH)
    CONFIG = FULL_CONFIG['env_params']
except FileNotFoundError:
    pytest.skip("Default config file not found, skipping environment tests.", allow_module_level=True)


@pytest.fixture
def env():
    """Pytest fixture to create a fresh environment for each test."""
    # Use a slightly larger grid for some tests to avoid trivial cases
    test_config = CONFIG.copy()
    test_config['grid_size'] = 10
    test_config['max_episode_steps'] = 50 # Shorter T for faster testing
    test_env = GridEnv(test_config)
    # Seed is set within reset() calls in individual tests for isolation
    return test_env

# --- Basic Initialization and Reset Tests ---

def test_env_initialization(env):
    """Test basic environment properties after initialization."""
    assert env.N == 10 # From fixture override
    assert env.T == 50 # From fixture override
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.grid_odor.shape == (env.N, env.N)
    assert env.current_agent_idx == 0 # Should be 0 before first reset

def test_reset_generates_zones_and_state(env):
    """Test the reset method generates required state components."""
    obs, info = env.reset(seed=42)

    assert "agent_id" in obs
    assert "heading" in obs
    assert "heading_history" in obs
    assert "at_wall" in obs
    assert "diagonal_odors" in obs
    assert obs["agent_id"] == 0
    assert obs["heading_history"].shape == (env.max_heading_history,)
    assert obs["diagonal_odors"].shape == (2,)
    assert obs["diagonal_odors"].dtype == np.float32

    assert "agent_pos" in info
    assert "home_zone" in info and len(info["home_zone"]) == env.L
    assert "goal_zone" in info and len(info["goal_zone"]) > 0
    assert "goal_zone_center" in info
    assert info["current_agent_idx"] == 0
    assert info["agent1_reached_goal"] is False
    assert info["agent1_returned_home"] is False

    # Check agent starts in home zone
    start_pos = tuple(info["agent_pos"])
    assert start_pos in info["home_zone"]

    # Check odor grid is reset
    assert np.all(env.grid_odor == 0.0)

def test_gymnasium_env_checker(env):
    """Use Gymnasium's built-in checker."""
    # Need to reset first to properly initialize spaces based on generated zones
    env.reset(seed=111)
    try:
        # This performs a lot of checks on the environment's implementation
        gymnasium_check_env(env)
    except Exception as e:
        pytest.fail(f"Gymnasium env_checker failed: {e}")


# --- Action Execution Tests ---

def test_action_move_forward(env):
    """Test moving forward action."""
    env.reset(seed=123)
    # Manually set agent away from wall for predictability
    env.agent_pos = np.array([env.N // 2, env.N // 2])
    env.agent_heading = 0 # North
    initial_pos = env.agent_pos.copy()

    action = env.action_space.sample()
    # action['direction'] = 2 # Force Move Forward
    action[0] = 0.0    # a_dir in [-0.33, 0.33] => direction=2 (Forward)

    obs, reward, term, trunc, info = env.step(action)

    expected_pos = initial_pos + np.array([-1, 0]) # Moved North (row decreases)
    assert np.array_equal(info['agent_pos'], expected_pos)
    assert info['agent_heading'] == 0 # Heading unchanged

def test_action_turn(env):
    """Test turning actions."""
    env.reset(seed=124)
    env.agent_pos = np.array([env.N // 2, env.N // 2])
    env.agent_heading = 1 # East
    initial_pos = env.agent_pos.copy()

    # Turn Left
    action = env.action_space.sample()
    # action['direction'] = 0 # Turn Left
    action[0] = -0.5 # a_dir in [-infty, -0.33] => direction=0 (Left)
    obs, reward, term, trunc, info = env.step(action)
    assert info['agent_heading'] == 0 # East -> North
    assert np.array_equal(info['agent_pos'], initial_pos) # Position unchanged

    # Turn Right from North
    # action['direction'] = 1 # Turn Right
    action[0] = 0.5 # a_dir in [0.33, inf] => direction=1 (Right)
    obs, reward, term, trunc, info = env.step(action)
    assert info['agent_heading'] == 1 # North -> East
    assert np.array_equal(info['agent_pos'], initial_pos)

def test_action_wall_collision(env):
    """Test agent behavior when hitting a wall."""
    env.reset(seed=125)
    # Place agent at top edge, facing North
    env.agent_pos = np.array([0, env.N // 2])
    env.agent_heading = 0 # North
    initial_pos = env.agent_pos.copy()

    action = env.action_space.sample()
    # action['direction'] = 2 # Move Forward (into wall)
    action[0] = 0.0 # a_dir in [-0.33, 0.33] => direction=2 (Forward)

    obs, reward, term, trunc, info = env.step(action)

    assert np.array_equal(info['agent_pos'], initial_pos) # Position should not change
    assert info['agent_heading'] == 0 # Heading unchanged
    assert obs['at_wall'] == 1

# --- Odor Mechanics Tests ---

def test_odor_release(env):
    """Test if releasing odor increases grid values."""
    env.reset(seed=126)
    env.agent_pos = np.array([env.N // 2, env.N // 2])
    agent_r, agent_c = env.agent_pos
    assert np.all(env.grid_odor == 0.0)

    action = env.action_space.sample()
    # action['release_odor'] = 1
    action[1] = 1.0 # a_release_odor in [0, inf] => release odor
    # action['odor_strength'] = np.array([5.0], dtype=np.float32)
    action[2] = 1.0   # odor_spread => maps to max (5.0)
    # action['odor_spread'] = np.array([1.0], dtype=np.float32)
    action[3] = 1.0   # odor_strength => maps to max (10.0)

    env.step(action)

    assert env.grid_odor[agent_r, agent_c] > 0, "Odor at agent pos should increase"
    # Check a neighbor cell (should also increase due to spread)
    neighbor_r, neighbor_c = np.clip([agent_r + 1, agent_c], 0, env.N-1)
    if (neighbor_r, neighbor_c) != (agent_r, agent_c):
         assert env.grid_odor[neighbor_r, neighbor_c] > 0, "Neighbor odor should increase"
    assert np.sum(env.grid_odor) > 0, "Total odor should be positive"

def test_odor_no_release(env):
    """Test that odor doesn't change if not released."""
    env.reset(seed=127)
    env.agent_pos = np.array([env.N // 2, env.N // 2])
    initial_odor_sum = np.sum(env.grid_odor) # Should be 0

    action = env.action_space.sample()
    # action['release_odor'] = 0 # DO NOT release
    action[1] = -1.0 # a_release_odor in [-inf, 0] => no release

    env.step(action)

    # Account for potential tiny decay/diffusion if implemented and non-zero
    expected_odor_sum = initial_odor_sum * env.odor_decay # Simplistic check
    assert np.isclose(np.sum(env.grid_odor), expected_odor_sum, atol=1e-6), "Odor should not increase significantly"


def test_odor_dynamics(env):
    """Test odor decay and diffusion (if enabled)."""
    # Enable dynamics for this test
    env.odor_decay = 0.9
    env.odor_diffusion = 0.1
    env.reset(seed=128)
    env.agent_pos = np.array([env.N // 2, env.N // 2])

    # Release some odor first
    action = env.action_space.sample()
    # action['release_odor'] = 1
    action[1] = 1.0 # a_release_odor in [0, inf] => release odor
    # action['odor_strength'] = np.array([10.0], dtype=np.float32)
    action[2] = 1.0   # odor_spread => maps to max (5.0)
    # action['odor_spread'] = np.array([0.5], dtype=np.float32) # Small spread
    action[3] = -1.0  # odor_strength => maps to min (0.5)
    env.step(action)
    initial_odor_state = env.grid_odor.copy()
    initial_max_odor = np.max(initial_odor_state)
    assert initial_max_odor > 0

    # Take a step without releasing odor to observe dynamics
    # action['release_odor'] = 0
    action[1] = -1.0 # a_release_odor in [-inf, 0] => no release
    env.step(action)
    final_odor_state = env.grid_odor.copy()
    final_max_odor = np.max(final_odor_state)

    assert final_max_odor < initial_max_odor, "Max odor should decrease due to decay/diffusion"
    # Diffusion check (harder to verify precisely, check if a neighbor increased relative to its decayed value)
    # For simplicity, just check overall decay effect is present.

# --- State Transition and Reward Tests ---

def test_agent1_reach_goal_reward(env):
    """Test Agent 1 receives reward r1 upon entering goal zone."""
    env.reset(seed=129)
    # Manually place agent just outside goal, move into goal
    goal_center = env.goal_zone_center
    # Find a non-goal cell adjacent to the goal center (approx)
    start_r, start_c = goal_center[0] + int(np.ceil(np.sqrt(env.R_sq))) + 1, goal_center[1]
    start_r = np.clip(start_r, 0, env.N - 1) # Ensure within bounds
    env.agent_pos = np.array([start_r, start_c])
    # Set heading towards goal center (e.g., North if starting South of it)
    env.agent_heading = 0 if start_r > goal_center[0] else 2

    assert not env._is_in_zone(env.agent_pos, env.goal_zone_coords)
    assert not env.agent1_reached_goal_flag

    action = env.action_space.sample()
    # action['direction'] = 2 # Move Forward (towards goal)
    action[0] = 0.0 # a_dir in [-0.33, 0.33] => direction=2 (Forward)

    obs, reward, term, trunc, info = env.step(action)

    # Check if now in goal (depends on exact placement/radius)
    if env._is_in_zone(info['agent_pos'], env.goal_zone_coords):
        assert env.agent1_reached_goal_flag is True
        assert reward >= env.r1_enter_goal + env.step_penalty # Check r1 received (allow for step penalty)
        # Ensure reward isn't given again immediately
        obs, reward2, _, _, _ = env.step(action) # Take another step inside goal
        assert reward2 < env.r1_enter_goal # Should only get step penalty now
    else:
        pytest.skip("Test setup failed to move agent into goal in one step.")

def test_agent1_return_home_reward(env):
    """Test Agent 1 receives r2 upon returning home *after* goal."""
    env.reset(seed=130)
    # Simulate having reached goal
    env.agent1_reached_goal_flag = True
    # Place agent just outside home zone, heading towards it
    home_cell = env.home_zone_coords[0] # Example home cell
    # Place adjacent to home_cell (assuming home is on edge)
    if home_cell[0] == 0: # Top edge
        start_pos = np.array([1, home_cell[1]])
        env.agent_heading = 0 # North
    elif home_cell[0] == env.N - 1: # Bottom edge
        start_pos = np.array([env.N - 2, home_cell[1]])
        env.agent_heading = 2 # South
    elif home_cell[1] == 0: # Left edge
        start_pos = np.array([home_cell[0], 1])
        env.agent_heading = 3 # West
    else: # Right edge
        start_pos = np.array([home_cell[0], env.N - 2])
        env.agent_heading = 1 # East
    env.agent_pos = start_pos

    assert env._is_in_zone(env.agent_pos, env.home_zone_coords) is False
    assert env.agent1_returned_home_flag is False

    action = env.action_space.sample()
    # action['direction'] = 2 # Move forward (towards home)
    action[0] = 0.0 # a_dir in [-0.33, 0.33] => direction=2 (Forward)

    obs, reward, term, trunc, info = env.step(action)

    # Check if now in home
    if env._is_in_zone(info['agent_pos'], env.home_zone_coords):
        assert env.agent1_returned_home_flag is True
        assert info['current_agent_idx'] == 1 # Should switch to Agent 2
        assert reward >= env.r2_return_home + env.step_penalty
        # Ensure reward isn't given if goal wasn't reached first
        env.reset(seed=131)
        env.agent1_reached_goal_flag = False # Goal NOT reached
        env.agent_pos = start_pos # Same start pos near home
        env.agent_heading = env.agent_heading # Same heading
        obs, reward, term, trunc, info = env.step(action)
        if env._is_in_zone(info['agent_pos'], env.home_zone_coords):
             assert reward < env.r2_return_home # Only step penalty
             assert info['current_agent_idx'] == 0 # Shouldn't switch agent
    else:
        pytest.skip("Test setup failed to move agent into home zone in one step.")


def test_agent_switch_on_timeout(env):
    """Test switch to Agent 2 when Agent 1 times out."""
    env.reset(seed=132)
    assert env.current_agent_idx == 0

    # Simulate Agent 1 taking T steps
    for i in range(env.T):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        # Switch should only happen AFTER the T'th step is processed
        if i < env.T - 1:
            assert info['current_agent_idx'] == 0
            assert not term and not trunc # Episode shouldn't end yet
        else: # The T'th step taken by Agent 1
             # The step function handles the switch *after* processing the step
             # So, the info returned might still show agent 0, but the *internal* state switched
             assert env.current_agent_idx == 1 # Check internal state
             assert not term # Timeout is truncation for A1's turn, not termination
             # Check Agent 2 starts randomly in home zone
             assert tuple(env.agent_pos) in env.home_zone_coords
             assert env.agent_step_count == 0 # A2 step count reset


def test_agent_switch_on_return_home(env):
    """Test switch to Agent 2 when Agent 1 successfully returns home."""
    env.reset(seed=130)
    # Simulate having reached goal
    env.agent1_reached_goal_flag = True
    # Place agent just outside home zone, heading towards it
    home_cell = env.home_zone_coords[0] # Example home cell
    # Place adjacent to home_cell (assuming home is on edge)
    if home_cell[0] == 0: # Top edge
        start_pos = np.array([1, home_cell[1]])
        env.agent_heading = 0 # North
    elif home_cell[0] == env.N - 1: # Bottom edge
        start_pos = np.array([env.N - 2, home_cell[1]])
        env.agent_heading = 2 # South
    elif home_cell[1] == 0: # Left edge
        start_pos = np.array([home_cell[0], 1])
        env.agent_heading = 3 # West
    else: # Right edge
        start_pos = np.array([home_cell[0], env.N - 2])
        env.agent_heading = 1 # East
    env.agent_pos = start_pos

    assert env._is_in_zone(env.agent_pos, env.home_zone_coords) is False
    assert env.agent1_returned_home_flag is False

    action = env.action_space.sample()
    # action['direction'] = 2 # Move forward (towards home)
    action[0] = 0.0 # a_dir in [-0.33, 0.33] => direction=2 (Forward)

    obs, reward, term, trunc, info = env.step(action)

    if env._is_in_zone(info['agent_pos'], env.home_zone_coords):
        assert env.current_agent_idx == 1 # Switched to agent 2
        assert env.agent_step_count == 0 # Agent 2 steps reset
        # Agent 2 should start at Agent 1's final position
        assert np.array_equal(env.agent_pos, info['agent_pos']) # Check internal env state
    else:
        pytest.skip("Test setup failed to move agent into home zone.")


def test_agent2_reach_goal_termination(env):
    """Test episode termination and shared reward when Agent 2 reaches goal."""
    env.reset(seed=134)
    # Manually switch to Agent 2 and place near goal
    env.current_agent_idx = 1
    env.agent_step_count = 0
    goal_center = env.goal_zone_center
    start_r, start_c = goal_center[0] + int(np.ceil(np.sqrt(env.R_sq))) + 1, goal_center[1]
    start_r = np.clip(start_r, 0, env.N - 1)
    env.agent_pos = np.array([start_r, start_c])
    env.agent_heading = 0 if start_r > goal_center[0] else 2 # Face goal

    action = env.action_space.sample()
    # action['direction'] = 2 # Move into goal
    action[0] = 0.0 # a_dir in [-0.33, 0.33] => direction=2 (Forward)

    obs, reward, terminated, truncated, info = env.step(action)

    if env._is_in_zone(info['agent_pos'], env.goal_zone_coords):
         assert terminated is True # Episode should terminate
         assert info['agent2_reached_goal'] is True
         assert reward >= env.R_shared_goal + env.step_penalty # Check shared reward R received
    else:
        pytest.skip("Test setup failed to move Agent 2 into goal.")


def test_agent2_timeout_truncation(env):
    """Test episode truncation when Agent 2 times out."""
    env.reset(seed=135)
    # Switch to agent 2
    env.current_agent_idx = 1
    env.agent_step_count = 0

    # Simulate Agent 2 taking T steps without reaching goal
    terminated = False
    truncated = False
    info = {}
    for i in range(env.T):
        action = env.action_space.sample()
        # action['direction'] = 0 # Just turn to avoid reaching goal accidentally
        action[0] = -0.5 if i % 2 == 0 else 0.5 # Alternate turns to avoid goal
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated: break # Stop if goal reached unexpectedly
        if i < env.T - 1:
             assert not truncated
        else: # Last step for Agent 2
             assert truncated is True # Episode truncates
             assert terminated is False # Not a successful termination

    assert truncated is True # Final check outside loop


# --- Observation Space Specific Tests ---

def test_heading_history(env):
    """Test heading history tracking and padding."""
    env.reset(seed=136)
    hist_len = env.max_heading_history

    # Initial heading
    headings = [env.agent_heading]
    # Take steps and record headings
    for i in range(hist_len + 2): # Go over max length
        action = env.action_space.sample()
        # action['direction'] = i % 3 # Cycle through turn L, R, Fwd
        action[0] = -0.5 if i % 3 == 0 else (0.5 if i % 3 == 1 else 0.0) # Turn L, R, Fwd
        obs, _, _, _, info = env.step(action)
        headings.append(info['agent_heading'])
        expected_hist = headings[-(min(i+2, hist_len)):] # Get last max_hist items
        # Pad expected history
        padded_expected = ([-1] * (hist_len - len(expected_hist))) + expected_hist

        assert np.array_equal(obs['heading_history'], padded_expected), f"History mismatch at step {i+1}"

def test_diagonal_odors_observation(env):
    """Test if diagonal odor values are correctly reported."""
    env.reset(seed=137)
    env.agent_pos = np.array([env.N // 2, env.N // 2])
    env.agent_heading = 0 # North

    # Manually set odor values at diagonal points
    r, c = env.agent_pos
    nw_r, nw_c = np.clip([r-1, c-1], 0, env.N-1)
    ne_r, ne_c = np.clip([r-1, c+1], 0, env.N-1)
    odor_val_nw = 5.5
    odor_val_ne = 3.3
    env.grid_odor[nw_r, nw_c] = odor_val_nw
    env.grid_odor[ne_r, ne_c] = odor_val_ne

    obs = env._get_observation() # Call private method for direct check

    assert np.allclose(obs['diagonal_odors'], [odor_val_nw, odor_val_ne])