import sys
import pygame
import numpy as np
import gymnasium as gym

# Step 0: Import images for the environment
# -------
# Paths to your images
agent1_img_path = r'Photos\Mario.png'
agent2_img_path = r'Photos\SuperMario.png'
goal_img_path = r"Photos\Goal.png"
obstacle_img_path = r'Photos\Bomb.png'
background_img_path = r'Photos\Brick.png'
wall_img_path = r'Photos\Fence.png'
hell_img_path = r'Photos\Radiator.png'
food_img_path = r'Photos\Food.png'

# Step 1: Define your own custom environment
# -------
class PadmEnv(gym.Env):
    """
        Initializes the custom environment for a grid-based game using Gymnasium and Pygame.

        Game Details:
        - The grid is 8x8 in size. The agent begins at position (1,0) and aims to reach the primary goal at position (7,8).
        - Each cell in the grid can be a safe spot, contain foods (reward state), bomb (obstacle states), hell state.

        Parameters:
        - grid_size (int, optional): The size of the grid. Default is 9.
        - goal_coordinates (tuple, optional): The coordinates of the goal state. Default is (7, 8).

        Functionality:
        - Sets up the grid size and cell size for the environment.
        - Initializes the state, reward, info, and done status.
        - Sets the coordinates for the goal, hell states, wall states, obstacle states, and food states.
        - Loads images for the agent, goal, hell states, walls, obstacles, and food.
        - Defines the action space and observation space for the environment.
        - Initializes the Pygame window and sets its caption.
        
        Key Features:
        - Safe Moves: Every valid move by the agent grants a +1 reward.
        - Goal Achievement: Reaching the goal rewards the agent with +10 points and ends the game.
        - Reward Collection: Picking up a power-up gives Mario a +5 reward and transforms him into Super Mario.
        - Hell state: If the agent hit the hell state, the game is Over.
    """
    
    def __init__(self, grid_size=9, goal_coordinates=(7, 8)) -> None:
        super(PadmEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 85
        self.state = None
        self.reward = 0
        self.info = {}
        self.done = False
        self.goal = np.array(goal_coordinates)
        self.hell_states = []
        self.wall_states = []
        self.obstacle_states = []
        self.food_states = []
        self.agent_is_SuperMario = False
        self.food_collection_score = 0

        # Set the Window title
        pygame.display.set_caption("Super Mario Game")

        # Load images
        self.Mario_image = pygame.transform.scale(pygame.image.load(agent1_img_path), (self.cell_size, self.cell_size))
        self.goal_image = pygame.transform.scale(pygame.image.load(goal_img_path), (self.cell_size, self.cell_size))
        self.hell_image = pygame.transform.scale(pygame.image.load(hell_img_path), (self.cell_size, self.cell_size))
        self.SuperMario_image = pygame.transform.scale(pygame.image.load(agent2_img_path), (self.cell_size, self.cell_size))
        self.background_image = pygame.transform.scale(pygame.image.load(background_img_path), (self.cell_size, self.cell_size))
        self.wall_image = pygame.transform.scale(pygame.image.load(wall_img_path), (self.cell_size, self.cell_size))
        self.obstacle_image = pygame.transform.scale(pygame.image.load(obstacle_img_path), (self.cell_size, self.cell_size))
        self.food_image = pygame.transform.scale(pygame.image.load(food_img_path), (self.cell_size, self.cell_size))

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size * self.grid_size, self.cell_size * self.grid_size))

    def reset(self):
        """
            Resets the environment to its initial state.

            Functionality:
            - Resets the agent's position to the starting location.
            - Resets the game status flags such as done and agent's transformation state.
            - Initializes the reward and info dictionary.
            - Calculates the initial distance to the goal.
            - Returns the initial state and info dictionary.
        """
        self.state = np.array([1, 0])
        self.done = False
        self.food = 0
        self.agent_is_SuperMario = False
        self.reward = self.food_collection_score

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0]) ** 2 +
            (self.state[1] - self.goal[1]) ** 2
        )

        return self.state, self.info

    def add_hell_states(self, hell_state_coordinates):
        """
        Adds hell states to the environment.
        """
        self.hell_states.append(np.array(hell_state_coordinates))

    def add_wall_states(self, wall_state_coordinates):
        """
        Adds wall states to the environment.
        """
        self.wall_states.append(np.array(wall_state_coordinates))

    def add_obstacle_states(self, obstacle_state_coordinates):
        """
        Adds obstacle states to the environment.
        """
        self.obstacle_states.append(np.array(obstacle_state_coordinates))

    def add_food_states(self, food_state_coordinates):
        """
        Adds food states to the environment.
        """
        self.food_states.append(np.array(food_state_coordinates))
    

    def step(self, action):
        """
            Executes a step in the environment based on the action taken.

            Parameters:
            - action (int): The action to take. 0=Up, 1=Down, 2=Right, 3=Left

            Returns:
            - tuple: The new state, the reward, whether the game is done, and additional info.

            Functionality:
            - Updates the agent's position based on the action taken.
            - Calculates the distance to the goal.
            - Checks for collisions with walls, obstacles, hell states, and food.
            - Applies rewards or penalties based on the agent's new position.
            - Manages agent transformation between normal Mario and Super Mario states.
            - Returns the new state, accumulated reward, done status, and additional info.
        """
        next_state = self.state.copy()

        # Agent movement
        if action == 0 and self.state[0] > 0:  # Up
            next_state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            next_state[0] += 1
        elif action == 2 and self.state[1] < self.grid_size - 1:  # Right
            next_state[1] += 1
        elif action == 3 and self.state[1] > 0:  # Left
            next_state[1] -= 1

        # Calculate distance to goal
        self.info["Distance to goal"] = np.sqrt(
            (next_state[0] - self.goal[0]) ** 2 +
            (next_state[1] - self.goal[1]) ** 2
        )

        # Check wall states
        if any(np.array_equal(next_state, wall) for wall in self.wall_states):
            next_state = self.state

        # Check Hell states
        elif any(np.array_equal(next_state, each_hell) for each_hell in self.hell_states):
            self.done = True
            self.reward += -5

        # Check whether the agent is Agent1 or Agent2 while hitting with hell state
        elif any(np.array_equal(next_state, each_obstacle) for each_obstacle in self.obstacle_states) and not self.agent_is_SuperMario:
            self.reward += -5
            self.agent_is_SuperMario = False
            self.state, self.info = self.reset()
            return self.state, self.reward, self.done, self.info
        
        # If Agent is in update form (SuperMario) then convert it into normal form (Mario)
        elif any(np.array_equal(next_state, each_obstacle) for each_obstacle in self.obstacle_states) and self.agent_is_SuperMario:
            self.reward += -5
            self.agent_is_SuperMario = False
            return self.state, self.reward, self.done, self.info  

        # Check food states
        elif any(np.array_equal(next_state, food) for food in self.food_states):
            self.agent_is_SuperMario = True
            self.reward += 5
            self.food_states = [food for food in self.food_states if not np.array_equal(next_state, food)]
            self.state = next_state
            self.food_collection_score += 5

        # Check goal condition
        elif np.array_equal(next_state, self.goal):
            self.reward = 10
            self.done = True

        else:  # Every other state
            self.reward += 1

        self.state = next_state

        return self.state, self.reward, self.done, self.info

    def render(self):
        """
            Renders the current state of the environment.

            Functionality:
            - Handles Pygame events to allow window closing.
            - Draws the background grid and grid lines.
            - Draws the hell states, food states, goal state, wall states, and obstacles on the grid.
            - Draws the agent, using different images based on its transformation state.
            - Updates the display with the newly drawn frame.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the Background
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.screen.blit(self.background_image, (x * self.cell_size, y * self.cell_size))

        # Draw grid Lines
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), grid, 1)

        # Draw Hell States
        for hell in self.hell_states:
            self.screen.blit(self.hell_image, (hell[1] * self.cell_size, hell[0] * self.cell_size))

        # Draw Food States
        for food in self.food_states:
            self.screen.blit(self.food_image, (food[1] * self.cell_size, food[0] * self.cell_size))

        # Draw the Goal State
        self.screen.blit(self.goal_image, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size))

        # Draw Wall States
        for wall in self.wall_states:
            self.screen.blit(self.wall_image, (wall[1] * self.cell_size, wall[0] * self.cell_size))

        # Draw Obstacles
        for obstacle in self.obstacle_states:
            self.screen.blit(self.obstacle_image, (obstacle[1] * self.cell_size, obstacle[0] * self.cell_size))

        # Draw the Agent
        if self.agent_is_SuperMario:
            self.screen.blit(self.SuperMario_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))
        else:
            self.screen.blit(self.Mario_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))

        pygame.display.flip()

    def close(self):
        """
        Closes the environment.
        """
        pygame.quit()

def main():
    env = PadmEnv()
    obstacle_states = [(4, 2), (1, 4), (3, 6), (6, 5)]
    wall_states = [(0, 0), (0, 8), (2, 0), (1, 8), (2, 8), (3, 0), (3, 8), (4, 0), (4, 8), (5, 0), (5, 8), (6, 0), (6, 8), (7, 0), (8, 0), (8, 8), (0, 1), (8, 1), (0, 2), (8, 2), (0, 3), (8, 3), (0, 4), (8, 4), (0, 5), (8, 5), (0, 6), (8, 6), (0, 7), (8, 7), (6, 2), (5, 6), (3, 4)]
    hell_states = [(4,4)]
    food_states = [(2,2),(6, 7), (7, 1), (1, 7)]

    for hell in hell_states:
        env.add_hell_states(hell)

    for wall in wall_states:
        env.add_wall_states(wall)

    for obstacle in obstacle_states:
        env.add_obstacle_states(obstacle)

    for food in food_states:
        env.add_food_states(food)

    observation, info = env.reset()
    print(f"Initial state: {observation}, Info: {info}")

    try:
        while True:
            env.render()
            keys = pygame.key.get_pressed()
            action = None
            if keys[pygame.K_UP]:
                action = 0
            elif keys[pygame.K_DOWN]:
                action = 1
            elif keys[pygame.K_RIGHT]:
                action = 2
            elif keys[pygame.K_LEFT]:
                action = 3

            if action is not None:
                observation, reward, done, info = env.step(action)
                print(f"Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}")

            if 'done' in locals() and done:
                print("Game Over!")
                break

            pygame.time.Clock().tick(9)

    except Exception as e:
        print(e)
    finally:
        env.close()

if __name__ == '__main__':
    main()
