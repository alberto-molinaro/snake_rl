from main import Game 
from stable_baselines3 import PPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import numpy as np
import pygame

class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, GameClass):
        super(SnakeEnv, self).__init__()
        self.game = GameClass()
        self.action_space = spaces.Discrete(4)

        # Update the observation space to match the shape (8,) of observations
        # Assuming the observation consists of the snake's head position (x, y), food position (x, y),
        # and a one-hot encoded vector for direction, totaling 8 elements.
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                                            high=np.array([self.game.grid_width, self.game.grid_height, 
                                                           self.game.grid_width, self.game.grid_height, 
                                                           1, 1, 1, 1]),
                                            dtype=np.float32)
        
    def reset(self):
        self.game.reset()  # Reset game to a starting state
        initial_observation = self.get_observation() 
        return initial_observation

    def step(self, action):
        obs, reward, done = self.game.step(action)
        self.render(mode='human') 
        return obs, reward, done, {}

    def get_observation(self):
        head_x, head_y = self.game.snake.get_head_position()
        food_x, food_y = self.game.food
        direction = self.game.snake.direction

        # One-hot encode the direction
        direction_encoded = [0, 0, 0, 0]  # Up, Down, Left, Right
        if direction == (0, -1):  # Up
            direction_encoded[0] = 1
        elif direction == (0, 1):  # Down
            direction_encoded[1] = 1
        elif direction == (-1, 0):  # Left
            direction_encoded[2] = 1
        elif direction == (1, 0):  # Right
            direction_encoded[3] = 1

        # Concatenate all parts of the observation
        observation = np.array([head_x, head_y, food_x, food_y] + direction_encoded, dtype=np.float32)
        return observation
    
    def render(self, mode='human'):
        if mode == 'human':
            # Clear the screen
            self.game.screen.fill((0, 0, 0))
            
            # Draw the snake
            for pos in self.game.snake.positions:
                pygame.draw.rect(self.game.screen, (0, 255, 0), pygame.Rect(pos[0] * self.game.grid_size, pos[1] * self.game.grid_size, self.game.grid_size, self.game.grid_size))
            
            # Draw the food
            pygame.draw.rect(self.game.screen, (255, 0, 0), pygame.Rect(self.game.food[0] * self.game.grid_size, self.game.food[1] * self.game.grid_size, self.game.grid_size, self.game.grid_size))
            
            # Update the display
            pygame.display.flip()
            
            # To keep the window responsive, process event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

def main():
    env = SnakeEnv(Game)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95)
    model.learn(total_timesteps=500000)

    # Save the model
    model.save("snake_model")

    # After training, evaluate the agent manually
    observation = env.reset()
    for _ in range(1000):  # Run for 1000 steps or choose a different number
        action, _states = model.predict(observation, deterministic=True)
        observation, rewards, done, info = env.step(action)
        env.render(mode='human')  # Render each step to observe the agent
        if done:
            observation = env.reset()

if __name__ == "__main__":
    main()