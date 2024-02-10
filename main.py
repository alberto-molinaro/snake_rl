import pygame
import sys
import random
import numpy as np

class Snake:
    def __init__(self, grid_width, grid_height):
        self.length = 1
        self.positions = [(5, 5)]  # Starting position
        self.direction = (0, 1)  # Moving right initially
        self.score = 0
        self.color = (0, 255, 0)  # Green
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.game_over = False

    def get_head_position(self):
        return self.positions[0]

    def turn(self, direction):
        if direction == 0 and self.direction != (0, 1):  # Up
            self.direction = (0, -1)
        elif direction == 1 and self.direction != (0, -1):  # Down
            self.direction = (0, 1)
        elif direction == 2 and self.direction != (1, 0):  # Left
            self.direction = (-1, 0)
        elif direction == 3 and self.direction != (-1, 0):  # Right
            self.direction = (1, 0)

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new_x = cur[0] + x
        new_y = cur[1] + y
        
        # Check for wall collisions
        if new_x < 0 or new_x >= self.grid_width or new_y < 0 or new_y >= self.grid_height:
            # End the game by signaling a game over condition
            self.game_over = True  
            return  # Stop moving since the game is over
        
        new = (new_x, new_y)
        
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset()  # Reset if the snake collides with itself
        else:
            self.positions.insert(0, new)  # Update the snake's position
            if len(self.positions) > self.length:
                self.positions.pop()  # Remove the last segment if the snake hasn't eaten food

    def reset(self):
        self.length = 1
        self.positions = [(5, 5)]
        self.direction = (0, 1)  # Reset to moving right
        self.score = 0

    def draw(self, surface, grid_size):
        for p in self.positions:
            r = pygame.Rect((p[0] * grid_size, p[1] * grid_size), (grid_size, grid_size))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, (93, 216, 228), r, 1)

class Game:
    def __init__(self):
        pygame.init()
        self.width = 400
        self.height = 400
        self.grid_size = 20
        self.grid_width = self.width // self.grid_size
        self.grid_height = self.height // self.grid_size
        self.bg_color = (0, 0, 0)
        self.food_color = (255, 0, 0)
        self.food = self.random_food()
        self.snake = Snake(self.grid_width, self.grid_height)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")

        # Rewards
        self.reward_eat_food = 10
        self.reward_death = -10
        self.reward_move = -0.1

    def random_food(self):
        return (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))

    def draw_food(self):
        r = pygame.Rect((self.food[0] * self.grid_size, self.food[1] * self.grid_size), (self.grid_size, self.grid_size))
        pygame.draw.rect(self.screen, self.food_color, r)

    def handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.snake.turn(0)
                elif event.key == pygame.K_DOWN:
                    self.snake.turn(1)
                elif event.key == pygame.K_LEFT:
                    self.snake.turn(2)
                elif event.key == pygame.K_RIGHT:
                    self.snake.turn(3)

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.handle_keys()
            _, reward, game_over = self.step(self.snake.direction) 
            
            if game_over:
                print("Game Over! Resetting...")
                self.reset() 
            
            self.screen.fill(self.bg_color)
            self.snake.draw(self.screen, self.grid_size)
            self.draw_food()
            pygame.display.flip()
            clock.tick(10)

    def step(self, action):
        # Translate the action into a direction change for the snake
        self.snake.turn(action)

        # Move the snake based on the current direction
        self.snake.move()

        # Initialize reward as the move reward
        reward = self.reward_move

        # Check if the game is over (either by hitting a wall or the snake itself)
        if self.snake.game_over:
            game_over = True
            reward = self.reward_death  # Assign the game over reward
        else:
            game_over = False
            # Check if the snake eats food
            if self.snake.get_head_position() == self.food:
                self.snake.length += 1
                self.snake.score += 1
                self.food = self.random_food()
                reward = self.reward_eat_food  # Assign the food reward

        # Return observation, reward, game_over status
        return self.get_observation(), reward, game_over
    

    def is_dead(self):
        head = self.snake.get_head_position()
        return (head in self.snake.positions[1:]) or \
               (head[0] < 0 or head[0] >= self.grid_width) or \
               (head[1] < 0 or head[1] >= self.grid_height)
    
    def get_observation(self):        
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food
        direction = self.snake.direction
        
        # One-hot encode the snake's direction
        direction_encoded = [0, 0, 0, 0]  # Up, Down, Left, Right
        if direction == (0, -1):
            direction_encoded[0] = 1
        elif direction == (0, 1):
            direction_encoded[1] = 1
        elif direction == (-1, 0):
            direction_encoded[2] = 1
        elif direction == (1, 0):
            direction_encoded[3] = 1
        
        # Concatenate all parts of the observation
        observation = [head_x, head_y, food_x, food_y] + direction_encoded

        # Convert the observation to a NumPy array
        observation_np = np.array(observation, dtype=np.float32)
        
        return observation_np
    
    def reset(self):
        # Reinitialize the game state to start a new episode
        self.snake = Snake(self.grid_width, self.grid_height)  # Create a new snake instance
        self.food = self.random_food()  # Place a new food item
        self.score = 0  # Reset score
        self.game_over = False

        # Return the initial observation
        return self.get_observation()

if __name__ == "__main__":
    game = Game()
    game.run()
