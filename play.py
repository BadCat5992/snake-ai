import pygame
import torch
import os
import math
import random
import numpy as np
import torch.nn as nn

# ==== Game Settings ====
GRID_SIZE = 64
CELL_SIZE = 10
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 15  # Spielgeschwindigkeit

# ==== Dueling DQN ====
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)

# ==== Snake Game ====
class SnakeGame:
    def __init__(self, render=True):
        self.render = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake AI - Play Mode")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)
        self.reset()

    def reset(self):
        self.snake = [(random.randint(10, GRID_SIZE-10), random.randint(10, GRID_SIZE-10))]
        self.direction = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        self.food = self.spawn_food()
        self.steps = 0
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            f = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if f not in self.snake:
                return f

    def check_collision(self, pos):
        x, y = pos
        return x < 0 or y < 0 or x >= GRID_SIZE or y >= GRID_SIZE or pos in self.snake

    def get_valid_actions(self):
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        valid = []
        for i, nd in enumerate(dirs):
            if (nd[0] + self.direction[0], nd[1] + self.direction[1]) == (0,0):
                continue
            new = (self.snake[0][0] + nd[0], self.snake[0][1] + nd[1])
            if not self.check_collision(new):
                valid.append(i)
        return valid

    def get_state(self):
        head = self.snake[0]
        x, y = head
        fx, fy = self.food
        dir_x, dir_y = self.direction
        dist_x = fx - x
        dist_y = fy - y
        euclid = math.hypot(dist_x, dist_y)

        front = (x + dir_x, y + dir_y)
        left = (x - dir_y, y + dir_x)
        right = (x + dir_y, y - dir_x)
        danger_front = self.check_collision(front)
        danger_left  = self.check_collision(left)
        danger_right = self.check_collision(right)

        def free_distance(dx, dy):
            dist = 0
            cx, cy = x, y
            while 0 <= cx+dx < GRID_SIZE and 0 <= cy+dy < GRID_SIZE and (cx+dx, cy+dy) not in self.snake:
                cx += dx
                cy += dy
                dist += 1
            return dist / GRID_SIZE

        state = [
            danger_front,
            danger_left,
            danger_right,
            dir_x, dir_y,
            dist_x / GRID_SIZE,
            dist_y / GRID_SIZE,
            euclid / (math.sqrt(2) * GRID_SIZE),
            free_distance(0, -1),
            free_distance(0, 1),
            free_distance(-1, 0),
            free_distance(1, 0)
        ]
        return torch.FloatTensor(state)

    def step(self, action):
        self.steps += 1
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        nd = dirs[action]
        if (nd[0] + self.direction[0], nd[1] + self.direction[1]) != (0,0):
            self.direction = nd
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        if self.check_collision(new_head) or self.steps > 300 * len(self.snake):
            return None, -100, True
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()

        if self.render:
            self._draw()
        return self.get_state(), 0, False

    def _draw(self):
        self.screen.fill((0,0,0))
        for i,(x,y) in enumerate(self.snake):
            color = (0,255,0) if i==0 else (0,200,0)
            pygame.draw.rect(self.screen, color, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        fx, fy = self.food
        pygame.draw.rect(self.screen, (255,0,0), (fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        self.screen.blit(self.font.render(f'Score: {self.score}', True, (255,255,255)), (10,10))
        pygame.display.flip()
        self.clock.tick(FPS)

# ==== Agent ====
class DQNAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_env = SnakeGame(render=False)
        input_dim = len(dummy_env.get_state())
        self.model = DuelingDQN(input_dim, 128, 4).to(self.device)
        self.load_checkpoint()

    def act(self, state):
        valid_actions = SnakeGame(render=False).get_valid_actions()
        st = state.to(self.device)
        with torch.no_grad():
            q_vals = self.model(st.unsqueeze(0)).squeeze().cpu().numpy()
        mask = np.full_like(q_vals, -np.inf)
        mask[valid_actions] = q_vals[valid_actions]
        return int(np.argmax(mask))

    def load_checkpoint(self):
        try:
            files = [f for f in os.listdir('checkpoints') if f.startswith('ckpt_')]
            if not files:
                raise Exception("Kein Checkpoint gefunden.")
            latest = max(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            path = os.path.join('checkpoints', latest)
            print(f"[📂] Lade Checkpoint: {path}")
            ck = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ck['model'])
            print(f"[✅] Modell geladen – Letzter Checkpoint: {latest}")
        except Exception as e:
            print(f"[❌] Fehler beim Laden des Checkpoints: {e}")
            exit()

# ==== Run ====
if __name__ == '__main__':
    game = SnakeGame(render=True)
    agent = DQNAgent()
    while True:
        state = game.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            action = agent.act(state)
            next_state, _, done = game.step(action)
            if next_state is None:
                break
            state = next_state
