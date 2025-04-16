# Two-Agent Sequential RL Grid Environment

This project implements a reinforcement learning pipeline for a custom 2D grid environment where two agents act sequentially to achieve individual and shared goals. It uses Gymnasium for the environment API and Stable-Baselines3 for RL algorithms (PPO/SAC).

## Features

-   **Custom Gymnasium Environment:** `custom_env/grid_env.py` defines the grid world with specific rules.
-   **Sequential Two-Agent Task:** Agent 1 must navigate to a goal and return home; Agent 2 starts afterwards and must reach the goal.
-   **Complex Observation Space:** Includes agent heading, heading history, wall detection, and local odor perception.
-   **Complex Action Space:** Agents control direction (turn/move) and odor release (toggle, strength, spread) simultaneously.
-   **Gaussian Odor Model:** Agents can release odor that spreads according to a Gaussian function.
-   **Configurable Parameters:** Environment size, zone definitions, rewards, agent steps, and RL hyperparameters are managed via YAML (`config/default_config.yaml`).
-   **Stable-Baselines3 Integration:** Uses PPO/SAC for training (`training/train.py`).
-   **Logging & Monitoring:** Integrated with TensorBoard and logs episode statistics (`logs/`).
-   **Visualization:** Includes utilities to plot the grid state, odor map, and agent trajectories (`utils/visualization.py`).
-   **Reproducibility:** Supports setting random seeds and logs configuration.
-   **Unit Tests:** Basic tests for the environment (`tests/test_environment.py`).
-   **Distributed Training Ready:** The modular structure and use of `VecEnv` facilitate scaling. For large-scale distributed training across multiple machines, consider adapting the pipeline to use Ray RLLib.

## Environment Details

-   **Grid:** NxN square grid.
-   **Home Zone:** A random line of length L on a random periphery. Regenerated each episode.
-   **Goal Zone:** A random circular area of radius R. Regenerated each episode.
-   **Agent 1:** Starts in Home Zone. Reward `r1` for reaching Goal Zone, `r2` for returning Home *after* reaching Goal.
-   **Agent 2:** Starts where Agent 1 finished (if returned home) or randomly in Home Zone (if Agent 1 failed). Must reach Goal Zone.
-   **Shared Reward:** Both agents receive reward `R` if Agent 2 successfully reaches the Goal Zone. (Note: In the current SB3 implementation, Agent 2 receives this reward directly in its step return. Crediting Agent 1 might require more advanced techniques or MARL frameworks).
-   **Observations:** `Dict` space containing `agent_id`, `heading`, `heading_history`, `at_wall`, `diagonal_odors`.
-   **Actions:** `Dict` space containing `direction`, `release_odor`, `odor_spread`, `odor_strength`.
-   **Episode End:** Agent 2 reaches the goal (terminated=True), or Agent 2 runs out of its T steps (truncated=True).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter Pygame issues during rendering/visualization, ensure you have its prerequisites installed for your OS.*

## Usage

### Training

Run the main training script, specifying the configuration file:

```bash
python training/train.py --config config/default_config.yaml