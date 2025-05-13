import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque

# Game Constants
GRID_SIZE = 64
CELL_SIZE = 10
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

# Deep Q-Network
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

# Snake Game
class SnakeGame:
    def __init__(self, render=False, human_play=False):
        self.render = render
        self.human_play = human_play
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            title = "Snake AI - Training" if not human_play else "Snake AI - Play Mode"
            pygame.display.set_caption(title)
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
        if self.render:
            self.draw()
        return self.get_state()
    
    def spawn_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if food not in self.snake:
                return food
    
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
    
    def check_collision(self, pos):
        x, y = pos
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return True
        return pos in self.snake
    
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
        if self.check_collision(new_head) or self.steps > 100*len(self.snake):
            return None, -10, True
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1
        if self.render:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, 0, True
        return self.get_state(), reward, False

# DQN Agent
class DQNAgent:
    def __init__(self, load_model=False):
        self.model = DQN(14, 128, 4)
        self.target_model = DQN(14, 128, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_episode = 0
        if load_model:
            self.load_checkpoint()
        else:
            self.update_target_model()
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([x[0] for x in batch])
        actions = torch.LongTensor([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch])
        next_states = torch.stack([x[3] for x in batch])
        dones = torch.FloatTensor([x[4] for x in batch])
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_checkpoint(self, episode):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),
            'episode': episode
        }
        torch.save(checkpoint, f"{self.checkpoint_dir}/checkpoint_{episode}.pt")
        print(f"[💾] Checkpoint gespeichert: Episode {episode}")
    
    def load_checkpoint(self, filename=None):
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_')]
        if not files:
            print("[🚀] Kein Checkpoint gefunden. Starte bei Episode 0.")
            return
        filename = max(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint = torch.load(f"{self.checkpoint_dir}/{filename}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = deque(checkpoint['memory'], maxlen=10000)
        self.start_episode = checkpoint['episode']
        print(f"[✅] Checkpoint geladen: {filename} | Starte bei Episode {self.start_episode + 1}")

def train():
    try:
        max_episodes = int(input("🟡 Wie viele Spiele (Episoden) willst du trainieren? ➤ "))
    except:
        print("⚠️ Ungültige Eingabe, verwende Standardwert: 1000")
        max_episodes = 1000

    env = SnakeGame(render=False)
    agent = DQNAgent(load_model=True)
    batch_size = 64
    save_interval = 100

    for episode in range(agent.start_episode, agent.start_episode + max_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if next_state is None:
                break
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            agent.train(batch_size)

        agent.update_target_model()
        if episode % save_interval == 0:
            agent.save_checkpoint(episode)

        print(f"[🎮] Episode: {episode+1} | Score: {env.score} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    train()

