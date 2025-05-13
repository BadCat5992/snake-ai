import pygame
import torch
import torch.nn as nn
import random
import os
import time

# ---- Game Constants ----
GRID_SIZE = 64
CELL_SIZE = 10
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

# ---- DQN Modell ----
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---- Snake Game ----
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI - Play Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.reset()
        
    def reset(self):
        start_x = random.randint(10, GRID_SIZE-10)
        start_y = random.randint(10, GRID_SIZE-10)
        self.snake = [(start_x, start_y)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.food = self.spawn_food()
        self.score = 0
        self.steps = 0
        self.draw()
        return self.get_state()
    
    def spawn_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if food not in self.snake:
                return food
    
    def check_collision(self, pos):
        x, y = pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE or pos in self.snake
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dx = food_x - head_x
        dy = food_y - head_y
        state = [
            head_x / GRID_SIZE,
            head_y / GRID_SIZE,
            dx / GRID_SIZE,
            dy / GRID_SIZE,
            self.direction[0],
            self.direction[1],
            int(self.check_collision((head_x + 1, head_y))),
            int(self.check_collision((head_x - 1, head_y))),
            int(self.check_collision((head_x, head_y + 1))),
            int(self.check_collision((head_x, head_y - 1))),
            int((head_x + 1, head_y) in self.snake),
            int((head_x - 1, head_y) in self.snake),
            int((head_x, head_y + 1) in self.snake),
            int((head_x, head_y - 1) in self.snake)
        ]
        return torch.FloatTensor(state)
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        for i, segment in enumerate(self.snake):
            x, y = segment
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(self.screen, color, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        fx, fy = self.food
        pygame.draw.rect(self.screen, (255, 0, 0), (fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(0)

    def step(self, action):
        self.steps += 1
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        new_dir = directions[action]
        if (new_dir[0] + self.direction[0], new_dir[1] + self.direction[1]) != (0, 0):
            self.direction = new_dir
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        if self.check_collision(new_head) or self.steps > 100 * len(self.snake):
            return None, -10, True
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1
        self.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, 0, True
        return self.get_state(), reward, False

# ---- Load Model ----
def load_latest_model():
    checkpoint_dir = "checkpoints"
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    if not files:
        raise Exception("❌ Kein Checkpoint gefunden!")
    latest = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    checkpoint = torch.load(f"{checkpoint_dir}/{latest}")
    print(f"[✅] Geladen: {latest}")
    model = DQN(14, 128, 4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ---- Play ----
def play():
    model = load_latest_model()
    game = SnakeGame()
    while True:
        state = game.reset()
        done = False
        while not done:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()
            next_state, _, done = game.step(action)
            if next_state is None:
                break
            state = next_state
            time.sleep(0.1)
        print(f"[🏁] Spiel beendet. Score: {game.score}")
        time.sleep(2)

if __name__ == "__main__":
    play()

