import os
import argparse
import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env

# Import custom components
from custom_env.grid_env import GridEnv
from utils.helpers import load_config, set_seeds, get_latest_checkpoint
from utils.visualization import plot_episode_trajectory, plot_grid_and_odor # For evaluation

def train(config_path, resume=False, eval_only=False):
    """Main training script."""

    # --- 1. Load Configuration & Setup ---
    config = load_config(config_path)
    env_config = config['env_params']
    model_config = config['model_params']
    train_config = config['training_params']

    set_seeds(train_config.get('seed', None))

    # Create unique log directory name
    run_name = f"{model_config['algorithm']}_{env_config['grid_size']}x{env_config['grid_size']}_{int(time.time())}"
    log_dir = os.path.join(train_config['log_dir'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # Save config for reproducibility
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        import yaml
        yaml.dump(config, f)

    # --- 2. Create Environment ---
    # Use VecEnv for parallelization
    n_envs = train_config.get('n_envs', 1) if not eval_only else 1
    env_kwargs = {'config': env_config}

    # Check environment compliance (optional but recommended)
    print("Checking environment...")
    check_env(GridEnv(**env_kwargs), warn=True)
    print("Environment check passed.")

    # Create vectorized environment
    if n_envs > 1:
        print(f"Using {n_envs} parallel environments (SubprocVecEnv).")
        # Use SubprocVecEnv for true parallelism, DummyVecEnv for sequential execution (debugging)
        vec_env = make_vec_env(lambda: GridEnv(**env_kwargs), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
        print("Using 1 environment (DummyVecEnv).")
        vec_env = make_vec_env(lambda: GridEnv(**env_kwargs), n_envs=1, vec_env_cls=DummyVecEnv)

    # Wrap with VecMonitor to automatically log episode statistics
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "monitor.csv"))


    # --- 3. Define RL Model ---
    algo_map = {"PPO": PPO, "SAC": SAC}
    if model_config['algorithm'] not in algo_map:
        raise ValueError(f"Unsupported algorithm: {model_config['algorithm']}")
    Algorithm = algo_map[model_config['algorithm']]

    # Handle policy kwargs (e.g., network architecture)
    policy_kwargs = model_config.get('policy_kwargs', None)
    if policy_kwargs:
        # Example of parsing activation function string if needed (SB3 usually handles this)
        # if 'activation_fn' in policy_kwargs:
        #     import torch.nn as nn
        #     act_fn_map = {'relu': nn.ReLU, 'tanh': nn.Tanh}
        #     policy_kwargs['activation_fn'] = act_fn_map.get(policy_kwargs['activation_fn'].lower(), nn.ReLU)
        print(f"Using policy_kwargs: {policy_kwargs}")


    model_path = None
    if resume or eval_only:
        # Try to find the *specific* run to resume from if `log_dir` points to it,
        # otherwise search the parent `train_config['log_dir']`
        search_dir = log_dir if os.path.exists(os.path.join(log_dir, "rl_model")) else train_config['log_dir']
        model_path = get_latest_checkpoint(search_dir) # Or specify exact path
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = Algorithm.load(model_path, env=vec_env, custom_objects={'policy_kwargs': policy_kwargs})
            # Reset learning rate if needed for fine-tuning, or use saved value
            # model.learning_rate = model_config['learning_rate']
        else:
            print("Warning: Could not find model to load. Starting fresh.")
            resume = False # Force fresh start if model not found
            model_path = None # Ensure we don't try loading later

    if not model_path and not eval_only: # Only create a new model if not loading and not eval_only
        print("Creating new model...")
        model = Algorithm(
            policy=model_config['policy'],
            env=vec_env,
            learning_rate=model_config['learning_rate'],
            n_steps=model_config.get('n_steps', 2048), # PPO specific
            batch_size=model_config.get('batch_size', 64), # PPO specific
            n_epochs=model_config.get('n_epochs', 10),     # PPO specific
            gamma=model_config['gamma'],
            gae_lambda=model_config.get('gae_lambda', 0.95), # PPO specific
            clip_range=model_config.get('clip_range', 0.2), # PPO specific
            ent_coef=model_config.get('ent_coef', 0.0),
            vf_coef=model_config.get('vf_coef', 0.5), # PPO specific
            max_grad_norm=model_config.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir, # Log TensorBoard data within the run's folder
            seed=train_config.get('seed', None), # Seed the algorithm too
            verbose=1 # Set to 1 for training updates, 0 for less output
            # Add SAC specific params here if using SAC (buffer_size, learning_starts, tau, train_freq etc.)
            # buffer_size=model_config.get('buffer_size', 1_000_000) # SAC specific
            # learning_starts=model_config.get('learning_starts', 100) # SAC specific
            # tau=model_config.get('tau', 0.005) # SAC specific
        )

    if eval_only:
         if not model_path:
             print("Error: Cannot run eval_only without a loaded model.")
             return
         print("Starting evaluation...")
         evaluate_policy(model, vec_env.envs[0], n_episodes=train_config.get('n_eval_episodes', 10), render=True)
         vec_env.close()
         return # Exit after evaluation

    # --- 4. Define Callbacks ---
    callbacks = []

    # Save checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_config['save_freq'] // n_envs, 1), # Adjust freq based on n_envs
        save_path=os.path.join(log_dir, "rl_model"),
        name_prefix="rl_model"
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback (optional but recommended)
    eval_env = GridEnv(**env_kwargs) # Create a single separate env for evaluation
    eval_env = gym.wrappers.Monitor(eval_env, os.path.join(log_dir, "eval")) # Log eval episodes too

    # Optional: Stop training if mean reward reaches a threshold
    # reward_threshold = config['training_params'].get('reward_threshold', None)
    # if reward_threshold:
    #     stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    #     eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback, # Trigger stop callback
    #                                best_model_save_path=os.path.join(log_dir, 'best_model'),
    #                                log_path=os.path.join(log_dir, 'eval'),
    #                                eval_freq=max(train_config['eval_freq'] // n_envs, 1),
    #                                n_eval_episodes=train_config['n_eval_episodes'],
    #                                deterministic=True, render=False, verbose=1)

    # Standard Eval Callback
    eval_callback = EvalCallback(eval_env,
                                   best_model_save_path=os.path.join(log_dir, 'best_model'),
                                   log_path=os.path.join(log_dir, 'eval_logs'),
                                   eval_freq=max(train_config['eval_freq'] // n_envs, 1),
                                   n_eval_episodes=train_config['n_eval_episodes'],
                                   deterministic=True, render=False, verbose=1)

    callbacks.append(eval_callback)

    # --- 5. Train the Agent ---
    print(f"Starting training for {train_config['total_timesteps']} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=train_config['total_timesteps'],
        callback=callbacks,
        log_interval=train_config['log_interval'], # Logs per rollout collection
        tb_log_name="run", # TensorBoard log name (under log_dir/run)
        reset_num_timesteps=not resume # If resuming, don't reset timestep count
    )
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # --- 6. Save Final Model ---
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # --- 7. Close Environments ---
    vec_env.close()
    eval_env.close() # Close the eval env explicitly

    print("Training complete.")
    print(f"Access TensorBoard logs with: tensorboard --logdir {train_config['log_dir']}")

def evaluate_policy(model, env, n_episodes=5, render=True):
    """Evaluates the trained policy and renders episodes."""
    print(f"\n--- Evaluating Policy for {n_episodes} episodes ---")
    total_rewards = []
    total_steps = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        trajectory_info = [info] # Store info for plotting trajectory

        # Store initial state for trajectory plot if needed
        start_home_zone = info.get('home_zone', [])
        start_goal_zone = info.get('goal_zone', [])

        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            trajectory_info.append(info)
            if render:
                try:
                     # Use env's render if it supports 'human' mode directly
                     env.render()
                     time.sleep(0.1) # Slow down rendering
                     # Or use Matplotlib visualization
                     # plot_grid_and_odor(info) # Needs modifications to work standalone
                except Exception as e:
                     print(f"Render error: {e}. Disabling render.")
                     render=False # Disable future renders if it fails

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={episode_steps}, Terminated={done}, Truncated={truncated}")
        print(f"  Agent 1 Reached Goal: {info.get('agent1_reached_goal', False)}, Returned Home: {info.get('agent1_returned_home', False)}")
        print(f"  Agent 2 Reached Goal: {info.get('agent2_reached_goal', False)}")

        # Plot trajectory for this episode
        if len(trajectory_info) > 1: # Check if episode ran
             plot_episode_trajectory(trajectory_info, env.N, start_home_zone, start_goal_zone,
                                     title=f"Episode {episode+1} Trajectory")


    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation Complete:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Steps: {np.mean(total_steps):.1f}")
    print("-" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent in Custom Grid Environment")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in log_dir")
    parser.add_argument("--eval", action="store_true", help="Load model and run evaluation only")
    parser.add_argument("--logdir", type=str, default=None, help="Specify exact log directory to resume from or evaluate (overrides config)")

    args = parser.parse_args()

    # Override log_dir from config if specified via command line (useful for resuming specific runs)
    if args.logdir:
        config = load_config(args.config)
        config['training_params']['log_dir'] = args.logdir
        # Overwrite the config file path to use the potentially modified one (not ideal, better to pass config dict)
        # Or directly pass the logdir to train function if refactored

    train(args.config, resume=args.resume, eval_only=args.eval)