import pygame
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import os

# ==== Configurations ====
GRID_SIZE    = 64
CELL_SIZE    = 10
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

USE_CNN      = False     # True: CNN-Input, False: engineered features
N_STEPS      = 3         # für N-Step Returns
SOFT_TAU     = 0.005     # soft target update
GRAD_CLIP    = 1.0
LOG_DIR      = 'runs'
FPS          = 100       # Training GUI speed

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# ==== Dueling DQN ====
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        if USE_CNN:
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, hidden_dim), nn.ReLU()
            )
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if USE_CNN:
            x = self.net(x)
        else:
            x = torch.relu(self.fc1(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)

# ==== Prioritized Replay mit N-step ====
class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1

    def push(self, transition):
        max_prio = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        prios = self.priorities if len(self.memory) == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha
        P = probs / probs.sum()
        idxs = np.random.choice(len(self.memory), batch_size, p=P)
        samples = [self.memory[i] for i in idxs]
        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        weights = (len(self.memory) * P[idxs]) ** (-beta)
        weights /= weights.max()
        return samples, idxs, torch.FloatTensor(weights)

    def update_priorities(self, idxs, td_errors):
        for i, err in zip(idxs, td_errors):
            self.priorities[i] = abs(err) + 1e-6

# ==== Snake Game ====
class SnakeGame:
    def __init__(self, render=False):
        self.render = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)
        self.reset()

    def reset(self):
        self.snake = [(random.randint(10, GRID_SIZE-10), random.randint(10, GRID_SIZE-10))]
        self.direction = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        self.food = self.spawn_food()
        self.score = 0
        self.steps = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            f = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if f not in self.snake:
                return f

    def check_collision(self, pos):
        x, y = pos
        return x<0 or y<0 or x>=GRID_SIZE or y>=GRID_SIZE or pos in self.snake

    def get_valid_actions(self):
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        valid = []
        for i, nd in enumerate(dirs):
            # Skip direct reverse
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

        # Richtung (als Vektor)
        dir_x, dir_y = self.direction

        # Abstand zum Futter
        dist_x = fx - x
        dist_y = fy - y
        euclid = math.hypot(dist_x, dist_y)

        # Felder vor/neben der Schlange (kannst du noch verfeinern)
        front = (x + dir_x, y + dir_y)
        left  = (x - dir_y, y + dir_x)
        right = (x + dir_y, y - dir_x)
        danger_front = self.check_collision(front)
        danger_left  = self.check_collision(left)
        danger_right = self.check_collision(right)

        # Sichtweite: wie viele freie Felder in jede Richtung
        def free_distance(dx, dy):
            dist = 0
            cx, cy = x, y
            while 0 <= cx+dx < GRID_SIZE and 0 <= cy+dy < GRID_SIZE and (cx+dx, cy+dy) not in self.snake:
                cx += dx
                cy += dy
                dist += 1
            return dist / GRID_SIZE  # Normalisiert

        state = [
            # Danger
            danger_front,
            danger_left,
            danger_right,

            # Blickrichtung
            dir_x, dir_y,

            # Relative Position Futter
            dist_x / GRID_SIZE,
            dist_y / GRID_SIZE,
            euclid / (math.sqrt(2) * GRID_SIZE),  # Normierte Distanz

            # Sichtweite
            free_distance(0, -1),  # oben
            free_distance(0, 1),   # unten
            free_distance(-1, 0),  # links
            free_distance(1, 0),   # rechts
        ]

        return torch.FloatTensor(state)



    def step(self, action):
        self.steps += 1
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        nd = dirs[action]
        if (nd[0] + self.direction[0], nd[1] + self.direction[1]) != (0,0):
            self.direction = nd

        prev_head = self.snake[0]
        new_head = (prev_head[0] + nd[0], prev_head[1] + nd[1])

        if self.check_collision(new_head) or self.steps > 300 * len(self.snake):
            return None, -100, True

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            reward = 15
        else:
            old_dist = math.hypot(prev_head[0]-self.food[0], prev_head[1]-self.food[1])
            reward = -0.01
            new_dist = math.hypot(new_head[0]-self.food[0], new_head[1]-self.food[1])
            if new_dist < old_dist:
                reward += 0.05
            self.snake.pop()

        if self.render:
            self._draw()
        return self.get_state(), reward, False

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
    def __init__(self, load_model=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = len(SnakeGame(render=False).get_state())
        self.model = DuelingDQN(self.input_dim, 128, 4).to(self.device)
        self.target = DuelingDQN(self.input_dim, 128, 4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = PrioritizedReplay(50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.n_buf = deque(maxlen=N_STEPS)
        self.writer = SummaryWriter(LOG_DIR)
        self.update_target()
        os.makedirs('checkpoints', exist_ok=True)
        if load_model:
            self.load_checkpoint()

    def act(self, state):
        valid_actions = SnakeGame().get_valid_actions()
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        st = state.to(self.device)
        with torch.no_grad():
            q_vals = self.model(st.unsqueeze(0)).squeeze().cpu().numpy()
        mask = np.full_like(q_vals, -np.inf)
        mask[valid_actions] = q_vals[valid_actions]
        return int(np.argmax(mask))

    def store(self, trans):
        self.n_buf.append(trans)
        if len(self.n_buf) < N_STEPS: return
        R = sum((self.gamma**i) * t.reward for i,t in enumerate(self.n_buf))
        s0, a0 = self.n_buf[0].state, self.n_buf[0].action
        sn, d = self.n_buf[-1].next_state, self.n_buf[-1].done
        self.memory.push(Transition(s0, a0, R, sn, d))

    def train(self, batch_size, frame_idx):
        if len(self.memory.memory) < batch_size: return
        batch, idxs, weights = self.memory.sample(batch_size)
        batch = Transition(*zip(*batch))
        states = torch.stack([s.to(self.device) for s in batch.state])
        next_states = torch.stack([s.to(self.device) for s in batch.next_state])
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        weights = weights.to(self.device)
        with torch.no_grad():
            next_acts = self.model(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_acts.unsqueeze(1)).squeeze()
        curr_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        target = rewards + (1 - dones) * (self.gamma**N_STEPS) * next_q
        td_errors = target - curr_q
        loss = (weights * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()
        for tp, mp in zip(self.target.parameters(), self.model.parameters()):
            tp.data.mul_(1 - SOFT_TAU)
            tp.data.add_(SOFT_TAU * mp.data)
        self.memory.update_priorities(idxs, td_errors.detach().cpu().numpy())
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self.writer.add_scalar('loss', loss.item(), frame_idx)
        self.writer.add_scalar('epsilon', self.epsilon, frame_idx)

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, ep):
        ck = {
            'model': self.model.state_dict(),
            'target': self.target.state_dict(),
            'opt': self.optimizer.state_dict(),
            'eps': self.epsilon,
            'memory': self.memory.memory
        }
        torch.save(ck, f'checkpoints/ckpt_{ep}.pt')
        print(f"[💾] Checkpoint gespeichert: checkpoints/ckpt_{ep}.pt")

    def load_checkpoint(self, load_path=None):
        try:
            files = [f for f in os.listdir('checkpoints') if f.startswith('ckpt_')]
            if not files:
                print("[⚠️] Kein Checkpoint gefunden.")
                return
            latest = max(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            path = os.path.join('checkpoints', latest)
            print(f"[📂] Lade Checkpoint: {path}")
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck['model'])
            self.target.load_state_dict(ck.get('target', ck['model']))
            self.optimizer.load_state_dict(ck['opt'])
            self.epsilon = ck['eps']
            self.memory.memory.clear()
            for tr in ck['memory']:
                self.memory.push(tr)
            print(f"[✅] Erfolgreich geladen | ε = {self.epsilon:.3f}")
        except Exception as e:
            print(f"[❌] Fehler beim Laden des Checkpoints: {e}")

# ==== Training Loop ====
def train():
    episodes   = int(input("Wie viele Episenoden? ➤ ") or 10000)
    env        = SnakeGame(render=False)
    agent      = DQNAgent(load_model=True)
    batch_size = 64
    frame_idx  = 0
    save_every = 100

    for ep in range(1, episodes+1):
        state = env.reset()
        done = False
        total_r = 0
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.act(state)
            nxt, r, done = env.step(action)
            if nxt is None:
                break
            agent.store(Transition(state, action, r, nxt, done))
            state = nxt
            total_r += r
            frame_idx += 1
            agent.train(batch_size, frame_idx)
        if ep % save_every == 0:
            agent.save_checkpoint(ep)
        print(f"[🎮] Ep {ep} | Score {env.score} | R {total_r:.1f} | ε {agent.epsilon:.3f}")

if __name__ == '__main__':
    train()

