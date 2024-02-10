from stable_baselines3 import PPO
from train import SnakeEnv
from main import Game
import time  # Import the time module

def play_snake():
    # Make sure to use the same environment setup as during training
    env = SnakeEnv(Game)
    
    # Load the model
    model = PPO.load("snake_model")

    # Reset the environment and get the initial observation
    observation = env.reset()
    
    # Run the loaded model to play Snake
    for _ in range(1000):  # Adjust the range for longer or shorter gameplay
        action, _states = model.predict(observation, deterministic=True)
        observation, rewards, done, info = env.step(action)
        env.render(mode='human')  # Visualize the game
        
        time.sleep(0.1)  # Add a delay of 0.1 seconds (adjust as needed)
        
        if done:
            observation = env.reset()

if __name__ == "__main__":
    play_snake()