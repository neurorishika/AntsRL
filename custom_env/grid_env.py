import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from scipy.stats import multivariate_normal
from collections import deque

class GridEnv(gym.Env):
    """
    Custom Grid Environment for Two Sequential Agents.

    Observation Space (Dict):
        - agent_id: Discrete(2) - 0 for Agent 1, 1 for Agent 2
        - heading: Discrete(4) - 0:N, 1:E, 2:S, 3:W
        - heading_history: Box(shape=(max_hist,), low=0, high=3) - Past headings padded with -1
        - at_wall: Discrete(2) - 0: No, 1: Yes
        - diagonal_odors: Box(shape=(2,), low=0.0) - Odor values at diagonal cells

    Action Space (Dict):
        - direction: Discrete(3) - 0: Turn Left, 1: Turn Right, 2: Move Forward
        - release_odor: Discrete(2) - 0: No, 1: Yes
        - odor_spread: Box(shape=(1,), low=0.1, high=5.0) - Sigma for Gaussian odor
        - odor_strength: Box(shape=(1,), low=0.1, high=10.0) - Peak value for Gaussian odor
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config):
        super().__init__()

        self.N = config['grid_size']
        self.L = config['home_zone_length']
        self.R_sq = config['goal_zone_radius']**2 # Store squared radius for efficiency
        self.T = config['max_episode_steps']
        self.max_heading_history = config['max_heading_history']
        self.odor_decay = config.get('odor_decay', 1.0) # Optional decay
        self.odor_diffusion = config.get('odor_diffusion', 0.0) # Optional diffusion

        # Rewards
        self.r1_enter_goal = config['rewards']['r1_enter_goal']
        self.r2_return_home = config['rewards']['r2_return_home']
        self.R_shared_goal = config['rewards']['R_shared_goal']
        self.step_penalty = config['rewards']['step_penalty']
        # Optional reward shaping terms
        # self.distance_reward_scale = config['rewards'].get('distance_reward_scale', 0.0)

        # Grid and Zones
        self.grid_odor = np.zeros((self.N, self.N), dtype=np.float32)
        self.home_zone_coords = []
        self.goal_zone_center = None
        self.goal_zone_coords = []

        # Agent States
        self.agent_pos = np.array([0, 0]) # Current agent's position (row, col)
        self.agent_heading = 0            # 0:N, 1:E, 2:S, 3:W
        self.heading_history = deque(maxlen=self.max_heading_history)
        self.agent1_start_pos = None
        self.agent1_end_pos = None
        self.agent1_returned_home_flag = False
        self.agent1_reached_goal_flag = False

        # Episode State
        self.current_agent_idx = 0 # 0 for Agent 1, 1 for Agent 2
        self.agent_step_count = 0  # Steps taken by the current agent in its turn
        self.total_steps = 0       # Total steps in the episode

        # Define action space
        self.action_space = spaces.Dict({
            "direction": spaces.Discrete(3), # 0: Left, 1: Right, 2: Forward
            "release_odor": spaces.Discrete(2),
            "odor_spread": spaces.Box(low=0.1, high=5.0, shape=(1,), dtype=np.float32),
            "odor_strength": spaces.Box(low=0.1, high=10.0, shape=(1,), dtype=np.float32)
        })

        # Define observation space
        self.observation_space = spaces.Dict({
            "agent_id": spaces.Discrete(2),
            "heading": spaces.Discrete(4),
            "heading_history": spaces.Box(low=-1, high=3, shape=(self.max_heading_history,), dtype=np.int8),
            "at_wall": spaces.Discrete(2),
            "diagonal_odors": spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float32)
            # Consider adding relative goal position if helpful for shaping
            # "relative_goal": spaces.Box(low=-self.N, high=self.N, shape=(2,), dtype=np.float32)
        })

        # For rendering
        self.window = None
        self.clock = None

    def _generate_home_zone(self):
        """Generates a random line of length L along a periphery."""
        self.home_zone_coords = []
        side = random.randint(0, 3) # 0: top, 1: right, 2: bottom, 3: left
        start = random.randint(0, self.N - self.L)
        if side == 0: # Top edge (row 0)
            self.home_zone_coords = [(0, c) for c in range(start, start + self.L)]
            self.initial_heading = 2 # Face South
        elif side == 1: # Right edge (col N-1)
            self.home_zone_coords = [(r, self.N - 1) for r in range(start, start + self.L)]
            self.initial_heading = 3 # Face West
        elif side == 2: # Bottom edge (row N-1)
            self.home_zone_coords = [(self.N - 1, c) for c in range(start, start + self.L)]
            self.initial_heading = 0 # Face North
        else: # Left edge (col 0)
            self.home_zone_coords = [(r, 0) for r in range(start, start + self.L)]
            self.initial_heading = 1 # Face East
        # Ensure agent starts facing away from the wall
        start_pos = random.choice(self.home_zone_coords)
        return np.array(start_pos), self.initial_heading

    def _generate_goal_zone(self):
        """Generates a random circular goal zone, robustly handling small grids."""
        # Padding: Ensure the center allows the radius R to fit mostly within bounds.
        # Let's remove the '+1' which seemed too restrictive.
        # Padding represents the minimum distance the center must be from an edge.
        padding = int(np.ceil(np.sqrt(self.R_sq)))

        # Calculate valid range for the center coordinates
        lower_bound = padding
        # The upper bound is N-1 (max index) minus the padding.
        upper_bound = self.N - 1 - padding

        if lower_bound > upper_bound:
            # If padding requirement makes the range invalid (grid too small for radius)
            # Default to placing the center in the absolute middle of the grid.
            # Use warnings for debugging is good practice.
            print(f"\nWarning: Grid size {self.N} might be too small for goal radius "
                  f"{np.sqrt(self.R_sq):.1f} with padding {padding}. Placing goal center near middle.\n")
            center_r = self.N // 2
            center_c = self.N // 2
            # Ensure middle point is still valid index if N is very small (e.g., N=1)
            center_r = np.clip(center_r, 0, self.N - 1)
            center_c = np.clip(center_c, 0, self.N - 1)

        else:
            # Choose a random center within the valid range
            center_r = random.randint(lower_bound, upper_bound)
            center_c = random.randint(lower_bound, upper_bound)

        self.goal_zone_center = np.array([center_r, center_c])

        # Efficiently calculate goal zone coordinates using numpy broadcasting
        rows, cols = np.ogrid[:self.N, :self.N] # Create index grids
        dist_sq = (rows - center_r)**2 + (cols - center_c)**2
        self.goal_zone_coords = list(zip(*np.where(dist_sq <= self.R_sq)))

        # --- Robust check for overlap with home zone ---
        # Convert to sets for efficient intersection check
        goal_coords_set = set(self.goal_zone_coords)
        home_coords_set = set(map(tuple, self.home_zone_coords)) # Ensure home coords are tuples

        max_retries = 15 # Limit retries to avoid potential infinite loops
        retry_count = 0
        while not goal_coords_set.isdisjoint(home_coords_set) and retry_count < max_retries:
            retry_count += 1
            print(f"Warning: Goal zone overlapped with home zone. Retrying generation ({retry_count}/{max_retries})...")

            # Regenerate center position
            if lower_bound > upper_bound:
                 center_r, center_c = self.N // 2, self.N // 2 # Stick to middle if grid small
                 center_r = np.clip(center_r, 0, self.N - 1)
                 center_c = np.clip(center_c, 0, self.N - 1)
                 # If even middle overlaps, we might be stuck if home/goal are large relative to N
                 # Break retry early if we can't change the center
                 if retry_count > 1: break
            else:
                 center_r = random.randint(lower_bound, upper_bound)
                 center_c = random.randint(lower_bound, upper_bound)

            self.goal_zone_center = np.array([center_r, center_c])
            # Recalculate goal coordinates
            dist_sq = (rows - center_r)**2 + (cols - center_c)**2
            self.goal_zone_coords = list(zip(*np.where(dist_sq <= self.R_sq)))
            goal_coords_set = set(self.goal_zone_coords)

        if retry_count == max_retries and not goal_coords_set.isdisjoint(home_coords_set):
            print(f"\nWarning: Could not find non-overlapping goal zone after {max_retries} retries. Proceeding with overlap.\n")
            # Accept the overlap in this case

        # Ensure goal_zone_coords is populated, even if empty list (e.g., R=0)
        if not hasattr(self, 'goal_zone_coords'):
             self.goal_zone_coords = []


        return self.goal_zone_center


    def _is_in_zone(self, pos, zone_coords):
        return tuple(pos) in zone_coords

    def _is_at_wall(self, pos):
        r, c = pos
        return r == 0 or r == self.N - 1 or c == 0 or c == self.N - 1

    def _get_diagonal_coords(self, pos, heading):
        """ Calculates the coordinates of the two diagonal cells in front."""
        r, c = pos
        if heading == 0: # North
            targets = [(r - 1, c - 1), (r - 1, c + 1)]
        elif heading == 1: # East
            targets = [(r - 1, c + 1), (r + 1, c + 1)]
        elif heading == 2: # South
            targets = [(r + 1, c - 1), (r + 1, c + 1)]
        else: # West
            targets = [(r - 1, c - 1), (r + 1, c - 1)]

        # Clamp coordinates to be within grid boundaries
        valid_coords = []
        for tr, tc in targets:
            vr = np.clip(tr, 0, self.N - 1)
            vc = np.clip(tc, 0, self.N - 1)
            valid_coords.append((vr, vc))
        return valid_coords

    def _get_observation(self):
        """Constructs the observation dictionary."""
        at_wall = self._is_at_wall(self.agent_pos)

        diag_coords = self._get_diagonal_coords(self.agent_pos, self.agent_heading)
        diag_odors = np.array([self.grid_odor[r, c] for r, c in diag_coords], dtype=np.float32)

        # Pad heading history
        hist = list(self.heading_history)
        padded_hist = ([-1] * (self.max_heading_history - len(hist))) + hist
        padded_hist = np.array(padded_hist, dtype=np.int8)

        obs = {
            "agent_id": self.current_agent_idx,
            "heading": self.agent_heading,
            "heading_history": padded_hist,
            "at_wall": int(at_wall),
            "diagonal_odors": diag_odors
        }
        # Optionally add relative goal position
        # if self.goal_zone_center is not None:
        #     relative_goal = self.goal_zone_center - self.agent_pos
        #     obs["relative_goal"] = relative_goal.astype(np.float32)
        # else: # Should not happen after reset, but safety first
        #     obs["relative_goal"] = np.zeros(2, dtype=np.float32)

        return obs

    def _apply_odor(self, pos, strength, spread):
        """Applies Gaussian odor centered at pos."""
        if strength <= 0: return

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(self.N), np.arange(self.N))
        pos_grid = np.stack([y, x], axis=-1) # Match (row, col) convention

        # Ensure covariance is positive definite; use diagonal covariance
        covariance = np.array([[max(spread, 0.1)**2, 0], [0, max(spread, 0.1)**2]])

        try:
            # Calculate Gaussian PDF values across the grid
            # Center the Gaussian at the agent's position `pos`
            rv = multivariate_normal(mean=pos, cov=covariance)
            gaussian_odor = rv.pdf(pos_grid)

            # Scale by strength and normalize (optional, adjust based on desired effect)
            # Normalization makes total odor added consistent regardless of spread
            gaussian_odor = strength * (gaussian_odor / np.max(gaussian_odor)) if np.max(gaussian_odor) > 0 else 0
            # Or simply scale by strength:
            # gaussian_odor *= strength * 10 # Scaling factor might need tuning

            # Add to existing odor grid
            self.grid_odor += gaussian_odor.astype(np.float32)
            self.grid_odor = np.maximum(0, self.grid_odor) # Ensure odor is non-negative

        except np.linalg.LinAlgError:
             print(f"Warning: Singular covariance matrix for odor spread {spread}. Skipping odor application.")
        except Exception as e:
            print(f"Error during odor application: {e}")


    def _apply_odor_dynamics(self):
        """Applies simple decay and diffusion to the odor grid."""
        # Decay
        self.grid_odor *= self.odor_decay

        # Diffusion (simple averaging with neighbors)
        if self.odor_diffusion > 0:
            diffused_odor = self.grid_odor.copy()
            for r in range(self.N):
                for c in range(self.N):
                    neighbors = []
                    if r > 0: neighbors.append(self.grid_odor[r-1, c])
                    if r < self.N - 1: neighbors.append(self.grid_odor[r+1, c])
                    if c > 0: neighbors.append(self.grid_odor[r, c-1])
                    if c < self.N - 1: neighbors.append(self.grid_odor[r, c+1])

                    if neighbors:
                        avg_neighbor_odor = np.mean(neighbors)
                        diffused_odor[r, c] = (1 - self.odor_diffusion) * self.grid_odor[r, c] + \
                                              self.odor_diffusion * avg_neighbor_odor
            self.grid_odor = diffused_odor

        self.grid_odor = np.maximum(0, self.grid_odor) # Ensure non-negative


    def reset(self, seed=None, options=None):
        """ Resets the environment to an initial state for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()
        # Existing reset code starts here
        self.current_step = 0

        # Reset environment state
        self.grid_odor.fill(0.0)
        self.agent_pos = np.array([0, 0])
        self.heading_history.clear()

        # Generate new zones for the episode
        self.agent1_start_pos, start_heading = self._generate_home_zone()
        self.goal_zone_center = self._generate_goal_zone()

        # Initialize Agent 1
        self.current_agent_idx = 0
        self.agent_step_count = 0
        self.total_steps = 0
        self.agent_pos = self.agent1_start_pos.copy()
        self.agent_heading = start_heading
        self.heading_history.append(self.agent_heading)
        self.agent1_reached_goal_flag = False
        self.agent1_returned_home_flag = False
        self.agent1_end_pos = None


        observation = self._get_observation()
        info = self._get_info()

        # Reset rendering if used
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """ Executes one time step for the current agent."""
        terminated = False
        truncated = False
        reward = self.step_penalty # Start with step penalty
        info = {}

        # --- 1. Parse Action ---
        direction_action = action["direction"]
        release_odor = action["release_odor"] == 1
        odor_spread = action["odor_spread"][0]
        odor_strength = action["odor_strength"][0]

        # --- 2. Execute Direction Control ---
        prev_pos = self.agent_pos.copy() # Store previous position for checks

        if direction_action == 0: # Turn Left
            self.agent_heading = (self.agent_heading - 1) % 4
        elif direction_action == 1: # Turn Right
            self.agent_heading = (self.agent_heading + 1) % 4
        elif direction_action == 2: # Move Forward
            dr, dc = [( -1,  0), (  0,  1), (  1,  0), ( 0, -1)][self.agent_heading]
            new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

            # Check boundaries (stay in place if hitting wall)
            if 0 <= new_r < self.N and 0 <= new_c < self.N:
                self.agent_pos = np.array([new_r, new_c])
            # else: reward -= 0.1 # Optional: penalty for hitting wall

        # Update heading history only after a move or turn
        self.heading_history.append(self.agent_heading)

        # --- 3. Execute Odor Release Control ---
        if release_odor:
            self._apply_odor(self.agent_pos, odor_strength, odor_spread)
            pass

        # --- 4. Apply Odor Dynamics (Decay/Diffusion) ---
        # Apply AFTER agent moves and potentially releases, before next observation
        self._apply_odor_dynamics()

        # --- 5. Update Step Counts ---
        self.agent_step_count += 1
        self.total_steps += 1

        # --- 6. Check Agent-Specific Goals and Calculate Rewards ---
        if self.current_agent_idx == 0: # Agent 1 Logic
            # Check if entered goal zone (and hasn't before)
            if not self.agent1_reached_goal_flag and self._is_in_zone(self.agent_pos, self.goal_zone_coords):
                reward += self.r1_enter_goal
                self.agent1_reached_goal_flag = True
                info['agent1_reached_goal'] = True

            # Check if returned home AFTER reaching goal (and hasn't before)
            if self.agent1_reached_goal_flag and \
               not self.agent1_returned_home_flag and \
               self._is_in_zone(self.agent_pos, self.home_zone_coords):
                reward += self.r2_return_home
                self.agent1_returned_home_flag = True
                self.agent1_end_pos = self.agent_pos.copy() # Store end position
                info['agent1_returned_home'] = True
                # Agent 1's turn ends successfully
                self._switch_to_agent2()

        else: # Agent 2 Logic
            # Check if reached goal zone
            if self._is_in_zone(self.agent_pos, self.goal_zone_coords):
                # SUCCESS: Agent 2 reached the goal
                reward += self.R_shared_goal # Agent 2 gets the shared reward directly
                terminated = True           # Episode ends successfully
                info['agent2_reached_goal'] = True
                # NOTE: Agent 1 doesn't get R directly in this step reward,
                # This might require adjustments in how training value is estimated
                # or use MARL frameworks if direct credit assignment is needed.

        # --- 7. Check Truncation Conditions ---
        if self.agent_step_count >= self.T:
            truncated = True
            if self.current_agent_idx == 0:
                # Agent 1 timed out, switch to Agent 2 (starting randomly)
                self.agent1_end_pos = None # Indicate failure to return home
                self._switch_to_agent2()
                # Truncation applies to Agent 1's turn, not the whole episode yet
                truncated = False # Reset truncated flag for Agent 2's turn
            # else: Agent 2 timed out - episode ends via truncation

        # --- 8. Get Next Observation ---
        # Observation is for the agent acting in the *next* step
        observation = self._get_observation()
        info.update(self._get_info()) # Add current state info

        # Render frame if in human mode
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    def _switch_to_agent2(self):
        """ Transitions control from Agent 1 to Agent 2. """
        self.current_agent_idx = 1
        self.agent_step_count = 0 # Reset step count for Agent 2
        self.heading_history.clear() # Reset history for Agent 2

        if self.agent1_returned_home_flag and self.agent1_end_pos is not None:
            # Start Agent 2 from Agent 1's end position
            self.agent_pos = self.agent1_end_pos.copy()
            # Determine starting heading (away from nearest wall) - simplified: use Agent 1's last valid heading
            # More robust: check position relative to walls/home zone boundary
            # For simplicity, let's assume Agent 1 ended facing *some* valid direction
            # If agent1_end_pos is on a wall, need careful heading logic.
            # Let's start facing North as a default if logic gets complex
            if self._is_at_wall(self.agent_pos):
                 # Basic check: if at top wall face S, left wall face E etc.
                 r, c = self.agent_pos
                 if r == 0: self.agent_heading = 2 # South
                 elif c == self.N -1 : self.agent_heading = 3 # West
                 elif r == self.N -1 : self.agent_heading = 0 # North
                 else: self.agent_heading = 1 # East
            # else keep Agent 1's final heading? Assume yes for now.
            # self.agent_heading = self.agent_heading # Keep last heading (already set)

        else:
            # Agent 1 failed (timed out or didn't return home), start Agent 2 randomly in home zone
            start_pos, start_heading = self._generate_home_zone() # Get a new valid start point
            self.agent_pos = start_pos
            self.agent_heading = start_heading # Face away from wall

        self.heading_history.append(self.agent_heading)


    def _get_info(self):
        """Returns dictionary with auxiliary information."""
        return {
            "agent_pos": self.agent_pos.copy(),
            "agent_heading": self.agent_heading,
            "current_agent_idx": self.current_agent_idx,
            "agent_step_count": self.agent_step_count,
            "total_steps": self.total_steps,
            "home_zone": self.home_zone_coords,
            "goal_zone_center": self.goal_zone_center,
            "goal_zone": self.goal_zone_coords,
            "agent1_reached_goal": self.agent1_reached_goal_flag,
            "agent1_returned_home": self.agent1_returned_home_flag,
        }

    def render(self):
        """ Renders the environment using Pygame. """
        if self.render_mode not in ["human", "rgb_array"]:
            return

        import pygame

        cell_size = 30 # Adjust for screen size
        margin = 5
        screen_size = self.N * cell_size + 2 * margin

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption("Grid Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((screen_size, screen_size))
        canvas.fill((255, 255, 255)) # White background

        # Draw Odor (Intensity as grayscale)
        max_odor = np.max(self.grid_odor) if np.max(self.grid_odor) > 0 else 1.0
        min_odor_display = 0.05 # Threshold to visualize faint odor
        for r in range(self.N):
            for c in range(self.N):
                 odor_val = self.grid_odor[r, c]
                 if odor_val > min_odor_display:
                    # Normalize odor for color intensity (log scale might be better)
                    intensity = min(200, int(200 * (odor_val / max_odor))) # Cap intensity
                    color = (255 - intensity, 255 - intensity, 255) # Light blue to darker blue
                    pygame.draw.rect(canvas, color, (margin + c * cell_size, margin + r * cell_size, cell_size, cell_size))


        # Draw Grid lines
        for x in range(self.N + 1):
            pygame.draw.line(canvas, (180, 180, 180), (margin + x * cell_size, margin), (margin + x * cell_size, margin + self.N * cell_size), 1)
            pygame.draw.line(canvas, (180, 180, 180), (margin, margin + x * cell_size), (margin + self.N * cell_size, margin + x * cell_size), 1)


        # Draw Home Zone
        for r, c in self.home_zone_coords:
            pygame.draw.rect(canvas, (0, 255, 0, 100), (margin + c * cell_size + 2, margin + r * cell_size + 2, cell_size - 4, cell_size - 4)) # Green, slightly smaller rect

        # Draw Goal Zone
        if self.goal_zone_center is not None:
             # Draw circle or just color cells
             for r, c in self.goal_zone_coords:
                 pygame.draw.rect(canvas, (0, 0, 255, 100), (margin + c * cell_size + 2, margin + r * cell_size + 2, cell_size - 4, cell_size - 4)) # Blue, slightly smaller rect
             # Draw center marker
             # center_px = (margin + self.goal_zone_center[1] * cell_size + cell_size // 2, margin + self.goal_zone_center[0] * cell_size + cell_size // 2)
             # pygame.draw.circle(canvas, (0, 0, 255), center_px, 5)


        # Draw Agent
        agent_color = (255, 0, 0) if self.current_agent_idx == 0 else (255, 165, 0) # Red for A1, Orange for A2
        agent_center_x = margin + self.agent_pos[1] * cell_size + cell_size // 2
        agent_center_y = margin + self.agent_pos[0] * cell_size + cell_size // 2
        agent_radius = cell_size // 3

        # Draw triangle for heading
        angle = np.radians(90 * self.agent_heading - 90) # 0:N -> -90deg, 1:E -> 0deg etc.
        p1 = (agent_center_x + agent_radius * np.cos(angle), agent_center_y + agent_radius * np.sin(angle))
        p2 = (agent_center_x + agent_radius * np.cos(angle + 2.3), agent_center_y + agent_radius * np.sin(angle + 2.3)) # approx 130 deg
        p3 = (agent_center_x + agent_radius * np.cos(angle - 2.3), agent_center_y + agent_radius * np.sin(angle - 2.3))
        pygame.draw.polygon(canvas, agent_color, [p1, p2, p3])

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
             return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            ) # Return as RGB array

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None