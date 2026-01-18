import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
import gymnasium as gym
from gymnasium import spaces
import math
import torch.nn.functional as F
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

SEQ_LENGTH = 4
# Hyperparameters
EPISODES = 400000
LR = 0.0001
BATCH_SIZE = 256
GAMMA = 0.999
MEMORY_SIZE = 100000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
NUM_HEADS = 3
MIN_MAX = [1.7, 3.8]

difficulty = 1.00
MODEL_ACTION = [3]
HEAL_PLUS = 0.0
STATES = 7
PREPOST_ENEMY_HEAL = [1, 1]
MY_DAMAGE = np.array([2.2])
ENEMY_DAMAGE = np.array([2.8, 3.2, 2.85, 2.95, 3.1, 2.45, 2.9, 2]) * difficulty
HEAL_ARCANE = 16
ENEMY_ARMOR = random.randint(0, 1)
timer = 0

def up_difficulty(up_coef=0.00):
    global difficulty, ENEMY_DAMAGE 
    difficulty += up_coef
    ENEMY_DAMAGE = np.array([3.3, 3.5]) 
    print(f"Updated ENEMY_DAMAGE: {ENEMY_DAMAGE}")

LOG_DETAILED = False
LOG_EPISODES_REMAINING = 0

def release_params():
    global HEAL_PLUS, ENEMY_ARMOR, HEAL_ARCANE, MY_DAMAGE, difficulty, ENEMY_DAMAGE
    HEAL_PLUS = random.uniform(-0.2, 0)
    ENEMY_ARMOR = 0
    HEAL_ARCANE = random.choice([15, 16, 17])
    
    ranges = np.linspace(MIN_MAX[0], MIN_MAX[1], NUM_HEADS + 1)
    head_idx = random.randint(0, NUM_HEADS - 1)
    min_dmg = ranges[head_idx]
    max_dmg = ranges[head_idx + 1]
    
    ENEMY_DAMAGE = np.random.uniform(min_dmg, max_dmg, 3)
    
    if random.randint(1, 4) == 1:
        alpha = 0.5
        beta = 0.5
        beta_values = np.random.beta(alpha, beta, 4)
        ENEMY_DAMAGE = MIN_MAX[0] + beta_values * (MIN_MAX[1] - MIN_MAX[0])
    
    ENEMY_DAMAGE = np.clip(ENEMY_DAMAGE, MIN_MAX[0], MIN_MAX[1])

class NumberGame(gym.Env):
    def __init__(self):
        super().__init__()

        self.STRATEGIES = {
            0: [1, 1, 2],
            1: [1, 2],
            2: [3],
            3: [4],
        }

        self.action_space = spaces.Discrete(len(self.STRATEGIES))
        self.observation_space = spaces.Box(
            low=np.array([0] * 7),
            high=np.array([1] * 7),
            dtype=np.float32
        )
        
        self.last_action = -1
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.heal_plus = HEAL_PLUS
        self.prepost_enemy_heal = PREPOST_ENEMY_HEAL
        self.my_damage = MY_DAMAGE.copy()
        self.enemy_damage = ENEMY_DAMAGE.copy()
        self.heal_arcane = HEAL_ARCANE
        self.enemy_armor = ENEMY_ARMOR
        self.model_action = MODEL_ACTION.copy()
        self.last_action = -1
        
        self.max_hp = 11
        self.old_hp = 10
        self.mana = 220
        self.max_mana = 220
        self.done = False
        self.enemy_xp = 1200
        self.max_enemy_xp = 11
        self.enemy_low_hp = False
        self.progress = 0
        self.kd_enemy_heal = 0
        self.heal = 1.84 
        self.resist = 0
        self.victory_rewarded = False
        self.defeat_processed = False
        self.hit_chance = random.uniform(0.45, 0.7)
        
        state = self._get_state()
        return state, {}
    
    def calculate_hp_mana_penalty_smooth(self, hp, mana):
        hp_max = self.old_hp
        mana_max = self.max_mana
        hp_normalized = max(0, min(1, hp / hp_max))
        mana_normalized = max(0, min(1, mana / mana_max))
        
        def calculate_single_penalty_exact(value_normalized):
            if value_normalized <= 0:
                return 0
            
            x = value_normalized
            key_points = [
                (0.000, 0),
                (0.001, -5),
                (0.010, -5),
                (0.100, -10),
                (0.400, -65),
                (0.500, -95),
                (1.000, -100)
            ]
            
            for i in range(len(key_points) - 1):
                x1, y1 = key_points[i]
                x2, y2 = key_points[i + 1]
                
                if x1 <= x <= x2:
                    t = (x - x1) / (x2 - x1)
                    penalty = y1 + (y2 - y1) * t
                    return penalty
            
            return -100
        
        hp_penalty = calculate_single_penalty_exact(hp_normalized)
        mana_penalty = calculate_single_penalty_exact(mana_normalized)
        
        total_penalty = (hp_penalty + mana_penalty * 1.35) 
        
        return total_penalty
    
    def _get_state(self):
        one_hot_last = np.zeros(4, dtype=np.float32)
        if self.last_action >= 0:
            one_hot_last[self.last_action] = 1.0
        
        delta_hp = (self.max_hp - self.old_hp) / 10.0
        
        return np.array([
            float(self.max_hp / 10.0),
            float(self.mana / self.max_mana),
            float(one_hot_last[0]),
            float(one_hot_last[1]),
            float(one_hot_last[2]),
            float(one_hot_last[3]),
            float(delta_hp),
        ], dtype=np.float32)
    
    def _get_context(self):
        return np.array([np.mean(ENEMY_DAMAGE), np.std(ENEMY_DAMAGE)])
    
    def _get_enemytype(self):
        a = np.mean(ENEMY_DAMAGE)
        ranges = np.linspace(MIN_MAX[0], MIN_MAX[1], NUM_HEADS + 1)
        digit = np.digitize(a, ranges) - 1
        return np.clip(digit, 0, NUM_HEADS - 1)

    def heavy_heal(self):
        if self.max_hp >= 10:
            self.max_hp = 10.5
            return -2  

        self.max_hp = min(self.max_hp + self.heal, 10)
        return 0

    def light_heal(self):
        if self.max_hp > 14:
            self.max_hp -= 0.5
        
        if self.mana <= 96:
            self.max_hp -= 1
            return -60

        self.max_hp = min(self.max_hp + self.heal + 0.05, 10)
        self.mana -= 96
        return 0

    def delayed_heal(self):
        self.timer_step(time_multiplier=random.uniform(0.1, 0.3))
        if self.max_hp >= 10:
            self.max_hp = 10.5
            return 5  
        self.max_hp = min(self.max_hp + self.heal, 10)
        return 10
    
    def damage(self, time_multiplayer):
        rage_coef = 1 + ((5 - self.max_hp) * 0.025)
        if rage_coef < 1:
            rage_coef = 1
        damage = (random.choice(self.enemy_damage) * rage_coef) * time_multiplayer
        self.max_hp -= damage
        return 0
        
    def draw(self):
        reward = self.calculate_hp_mana_penalty_smooth(self.max_hp, self.mana)

        return reward
    
    def timer_step(self, is_model_damage=True, time_multiplier=1.0):
        reward = 0
        self.progress += time_multiplier
        self.old_hp = self.max_hp
        
        damage_multiplier = 0.70 if is_model_damage else 1
        damage = random.choice(self.my_damage) * damage_multiplier * time_multiplier
        self.enemy_xp -= damage
        self.damage(time_multiplier)
 
        hp_regen = random.uniform(0.145, 0.20) * 1.5 if self.resist > 0 else random.uniform(0.06, 0.08)
        self.max_hp = min(self.max_hp + hp_regen * time_multiplier, 10)
        self.resist = max(self.resist - time_multiplier, 0)
        
        if self.kd_enemy_heal > 0:
            self.kd_enemy_heal -= time_multiplier    
        
        if self.enemy_xp <= 7 + self.prepost_enemy_heal[1]:
            if self.enemy_xp <= 3 + self.prepost_enemy_heal[0] and self.kd_enemy_heal <= 0:
                self.enemy_xp = min(self.enemy_xp + 3, self.max_enemy_xp)
                self.kd_enemy_heal = 3
                self.max_hp += 0.5 * time_multiplier
            elif random.randint(0, 2) == 2 and self.kd_enemy_heal <= 0:
                self.enemy_xp = min(self.enemy_xp + 2.8, self.max_enemy_xp)
                self.kd_enemy_heal = 3
                self.max_hp += 0.5 * time_multiplier
        
        self.mana = min(self.mana + self.heal_arcane * time_multiplier, self.max_mana)
        
        return reward

    def step(self, action):
        assert action in self.STRATEGIES, f"Invalid action: {action}. Must be 0-3."

        self.defeat_processed = False
        reward = 0
        
        old_hp = self.max_hp
        if self.max_hp < 5:
            reward -= (5 - self.max_hp) * 4
        
        for pre_action in self.STRATEGIES[action]:
            if pre_action == 1:
                reward += self.light_heal()
            elif pre_action == 2:
                reward += self.heavy_heal()
            elif pre_action == 3:
                reward += self.delayed_heal()
            elif pre_action == 4:
                reward += self.draw()
                self.done = True
        
        time_multiplier = 1.0
        
        if action in [0, 1, 2]:
            reward += self.timer_step(is_model_damage=True, time_multiplier=time_multiplier)
            self.enemy_xp += 0.2 * time_multiplier
        
        if action in [2, 3]:
            self.enemy_xp -= 0.1 * time_multiplier
            self.resist = 2
            reward += 3
        if action in [3]:
            reward += self.timer_step(is_model_damage=True, time_multiplier=time_multiplier)
        hp_change = self.max_hp - old_hp
        if hp_change > 0:
            reward += hp_change * 5
        
        reward += self.progress * 0.3
        
        if self.max_hp <= 0 and not self.defeat_processed:
            self.done = True
            self.defeat_processed = True
            reward = -500 + (self.progress * 8)
        
        if self.progress >= 65 and not self.done:
            self.done = True
            reward = 750 + (self.max_hp * 12)
            self.victory_rewarded = True
        
        reward = np.clip(reward, -600, 400)
        
        self.last_action = action  # Важно: сохраняем последнее действие
        next_state = self._get_state()
        
        return next_state, reward, self.done, False, {}


class WorldModel():
    def __init__(self,maxlen,batch_size,input_size,hidden_size,output_size,lr):
        self.model = self.MLP(input_size,hidden_size,output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer= self.Buffer(maxlen,batch_size=batch_size)
        self.buffer_batch_size = batch_size
#------------------sub classes start -------------------------------------------------------------------
    class MLP(nn.Module):
        def __init__(self, input_size=2, hidden_size=64, output_size=1):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
    
        def forward(self, x):
            return self.model(x)
    

    class Buffer(): 
      def __init__(self,maxlen,batch_size):
             self.replay_buffer = deque(maxlen=maxlen)
             self.maxlen= maxlen
             self.batch_size= batch_size
      def add(self,state,next_state):
            self.replay_buffer.append({'state':state,
                                        'next_state': next_state})
      def clean(self):
          self.replay_buffer=deque(self.maxlen)

      def __len__(self):
          return len(self.replay_buffer) 
         
      def sample(self):
          batch= []
          for i in range(self.batch_size):
              idx = random.randint(0, len(self.replay_buffer) - 1)
              sample = self.replay_buffer[idx]
              batch.append(sample)
          return batch  
#------------------sub classes start -------------------------------------------------------------------
    def train(self,num_epochs):
        if self.buffer.__len__() >= self.buffer_batch_size:
            loss_buffer= []
            for i in range(num_epochs):   
                batch = self.buffer.sample()
                # Векторизованная обработка батча
                states = torch.stack([torch.tensor(item["state"], dtype=torch.float32) for item in batch])
                next_states = torch.stack([torch.tensor(item["next_state"], dtype=torch.float32) for item in batch])

                self.optimizer.zero_grad()
                outputs=  self.model(states)
                loss= self.criterion(outputs,next_states)
                loss.backward()
                self.optimizer.step()
                mean_loss= np.mean(np.array(loss.item()))
                loss_buffer.append(mean_loss)    
            return np.mean(loss_buffer)    
        return -1    

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

class GatedDQN(nn.Module):
    """
    Gated Network с использованием истории состояний
    """
    def __init__(self, input_size, output_size=4, seq_length=SEQ_LENGTH):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_length = seq_length
        
        # Gate network: анализирует историю состояний
        self.gate_network = nn.Sequential(
            nn.Linear(seq_length * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # [0, 1] - gate values
        )
        
        # Базовые веса для первого слоя
        self.base_weight = nn.Parameter(torch.Tensor(128, input_size))
        self.base_bias = nn.Parameter(torch.Tensor(128))
        
        # Инициализация
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.base_bias, -bound, bound)
        
        # Основная сеть
        self.main_network = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(32, 16),
            nn.ReLU(),
            NoisyLinear(16, output_size)
        )
        
        self.layer_norm = nn.LayerNorm(128)
        
    def forward(self, x, history=None):
        batch_size = x.shape[0]
        
        # Если история не предоставлена, создаём её
        if history is None:
            history = x.repeat(1, self.seq_length)
        
        # Gate вычисляет коэффициенты модуляции
        gate_values = self.gate_network(history)  # [batch_size, 32]
        
        # Преобразуем gate values в modulation factors
        # Используем среднее по gate каналам для простоты
        modulation_factor = 0.7 + 0.3 * gate_values.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Модулируем веса
        modulated_weight = self.base_weight * modulation_factor.unsqueeze(2)  # [batch_size, 128, input_size]
        
        # Первый слой с modulated весами
        x_unsqueezed = x.unsqueeze(1)  # [batch_size, 1, input_size]
        x_gated = torch.bmm(x_unsqueezed, modulated_weight.transpose(1, 2)).squeeze(1)
        x_gated = x_gated + self.base_bias
        x_gated = F.relu(x_gated)
        x_gated = self.layer_norm(x_gated)
        
        # Основная сеть
        features = self.main_network(x_gated)
        
        # Dueling DQN
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, gate_values
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class ImpressionBuffer:
    def __init__(self, seq_len, states=STATES):
        self.seq_len = seq_len
        self.states = states
        self.reset()
    
    def reset(self):
        self.buffer = deque(maxlen=self.seq_len)
        self.buffer.extend([[0.0] * self.states for _ in range(self.seq_len)])
    
    def add(self, state):
        self.buffer.append(state)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_sequence(self):
        """Возвращает текущую последовательность (для основного входа)"""
        # Последнее состояние
        if len(self.buffer) > 0:
            return self.buffer[-1]
        return [0.0] * self.states
    
    def get_history_for_gate(self):
        """Возвращает всю историю для gate network"""
        return np.concatenate(self.buffer).tolist()

class Agent:
    def __init__(self, state_size=STATES, action_size=4, n_step=3):
        self.state_size = state_size
        self.action_size = action_size
        self.n_step = n_step
        
        # Используем GatedDQN
        self.policy_net = GatedDQN(state_size, action_size).to(device)
        self.target_net = GatedDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=LR,
            weight_decay=0.0001,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=50,
            min_lr=1e-6
        )
        
        # Упрощённая инициализация Replay Buffer
        self.memory = []
        self.priorities = []
        self.max_priority = 1.0
        
        # Буфер для снов (dreams)
        self.dream_memory = []
        self.dream_priorities = []
        self.dream_max_priority = 1.0
        
        self.n_step_buffer = deque(maxlen=n_step)
        self.epsilon = EPS_START
        
        self.training_losses = []
        self.best_reward = -float('inf')
        self.target_update_counter = 0
        
        self.gate_activations = []
        self.sample_probabilities = []
        
        # Флаг для использования снов
        self.use_dreams = False

    def act(self, state, history):
        """Выбор действия с использованием gated network"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            history_tensor = torch.tensor(history, device=device, dtype=torch.float32).unsqueeze(0)
            
            q_values, gate_values = self.policy_net(state_tensor, history_tensor)
            
            # Сохраняем для анализа
            if random.random() < 0.01:
                self.gate_activations.append(gate_values.cpu().numpy())
            
            # Epsilon-greedy
            if random.random() < self.epsilon:
                return random.randrange(self.action_size), gate_values.cpu().numpy()
            else:
                return q_values.argmax().item(), gate_values.cpu().numpy()
    
    def add_to_memory(self, transition, priority=None, is_dream=False):
        """Добавляем transition в память с приоритетом"""
        if is_dream:
            memory = self.dream_memory
            priorities = self.dream_priorities
            max_priority = self.dream_max_priority
        else:
            memory = self.memory
            priorities = self.priorities
            max_priority = self.max_priority
        
        if priority is None:
            priority = max_priority
        
        memory.append(transition)
        priorities.append(priority)
        
        # Поддерживаем максимальный размер
        if is_dream:
            max_size = MEMORY_SIZE // 2  # Память снов меньше
        else:
            max_size = MEMORY_SIZE
        
        if len(memory) > max_size:
            # Удаляем самый старый элемент
            del memory[0]
            del priorities[0]
        
        # Обновляем максимальный приоритет
        if is_dream:
            self.dream_max_priority = max(self.dream_max_priority, priority)
        else:
            self.max_priority = max(self.max_priority, priority)
    
    def sample_from_memory(self, batch_size, use_dreams=False):
        """Семплируем batch с учётом приоритетов"""
        if use_dreams:
            memory = self.dream_memory
            priorities = self.dream_priorities
        else:
            memory = self.memory
            priorities = self.priorities
        
        if len(memory) < batch_size:
            return None, None
        
        # Преобразуем приоритеты в вероятности
        priorities = np.array(priorities, dtype=np.float32)
        probs = priorities / priorities.sum()
        
        # Выбираем индексы
        indices = np.random.choice(len(memory), size=batch_size, p=probs, replace=False)
        
        # Собираем batch
        batch = [memory[i] for i in indices]
        
        # Вычисляем importance sampling weights
        weights = (len(memory) * probs[indices]) ** -0.4
        weights = weights / weights.max()
        
        # Конвертируем в тензоры
        states = torch.stack([t['state'] for t in batch]).to(device)
        actions = torch.tensor([t['action'] for t in batch], device=device, dtype=torch.long)
        rewards = torch.tensor([t['reward'] for t in batch], device=device, dtype=torch.float32)
        next_states = torch.stack([t['next_state'] for t in batch]).to(device)
        dones = torch.tensor([t['done'] for t in batch], device=device, dtype=torch.bool)
        histories = torch.stack([t['history'] for t in batch]).to(device)
        next_histories = torch.stack([t['next_history'] for t in batch]).to(device)
        is_weights = torch.tensor(weights, device=device, dtype=torch.float32)
        
        return (states, actions, rewards, next_states, dones, histories, next_histories, is_weights), indices
    
    def update_priorities(self, indices, td_errors, use_dreams=False):
        """Обновляем приоритеты для выбранных индексов"""
        if use_dreams:
            priorities = self.dream_priorities
            max_priority = self.dream_max_priority
        else:
            priorities = self.priorities
            max_priority = self.max_priority
        
        td_errors_np = np.abs(td_errors.cpu().detach().numpy()) + 1e-6
        
        for idx, td_error in zip(indices, td_errors_np):
            if idx < len(priorities):
                priorities[idx] = float(td_error)
        
        # Обновляем максимальный приоритет
        if len(td_errors_np) > 0:
            if use_dreams:
                self.dream_max_priority = max(self.dream_max_priority, float(td_errors_np.max()))
            else:
                self.max_priority = max(self.max_priority, float(td_errors_np.max()))
    
    def store_transition(self, state, action, reward, next_state, done, history, next_history, is_dream=False):
        """Сохраняем transition в n-step buffer"""
        # Создаём transition
        transition = {
            'state': torch.tensor(state, device=device, dtype=torch.float32),
            'action': action,
            'reward': reward,
            'next_state': torch.tensor(next_state, device=device, dtype=torch.float32),
            'done': done,
            'history': torch.tensor(history, device=device, dtype=torch.float32),
            'next_history': torch.tensor(next_history, device=device, dtype=torch.float32)
        }
        
        # Для снов сохраняем напрямую
        if is_dream:
            self.add_to_memory(transition, priority=self.dream_max_priority, is_dream=True)
            return
        
        self.n_step_buffer.append(transition)
        
        # Формируем n-step transition только когда буфер полон
        if len(self.n_step_buffer) == self.n_step:
            n_step_return = 0.0
            final_done = False
            
            # Вычисляем n-step return
            for i in range(self.n_step - 1, -1, -1):
                trans = self.n_step_buffer[i]
                n_step_return = trans['reward'] + GAMMA * n_step_return * (1.0 if not trans['done'] else 0.0)
                if trans['done']:
                    final_done = True
                    break
            
            first_trans = self.n_step_buffer[0]
            last_trans = self.n_step_buffer[-1]
            
            n_step_transition = {
                'state': first_trans['state'],
                'action': first_trans['action'],
                'reward': n_step_return,
                'next_state': last_trans['next_state'],
                'done': final_done or last_trans['done'],
                'history': first_trans['history'],
                'next_history': last_trans['next_history']
            }
            
            # Добавляем в память с начальным приоритетом
            self.add_to_memory(n_step_transition, priority=self.max_priority)
    
    def learn(self, dream_batch_size=0):
        """Шаг обучения с возможностью использования снов"""
        total_batch_size = BATCH_SIZE
        
        if dream_batch_size > 0 and len(self.dream_memory) >= dream_batch_size:
            # Используем смесь реальных данных и снов
            real_batch_size = total_batch_size - dream_batch_size
            dream_batch_size = dream_batch_size
        else:
            # Используем только реальные данные
            real_batch_size = total_batch_size
            dream_batch_size = 0
        
        # Проверяем, достаточно ли реальных данных
        if len(self.memory) < real_batch_size:
            return 0
        
        # Сбрасываем шум
        self.policy_net.reset_noise()
        
        # Семплируем реальные данные
        real_batch_data, real_indices = self.sample_from_memory(real_batch_size, use_dreams=False)
        
        # Если есть сны, семплируем их
        if dream_batch_size > 0:
            dream_batch_data, dream_indices = self.sample_from_memory(dream_batch_size, use_dreams=True)
            
            # Объединяем батчи
            states = torch.cat([real_batch_data[0], dream_batch_data[0]], dim=0)
            actions = torch.cat([real_batch_data[1], dream_batch_data[1]], dim=0)
            rewards = torch.cat([real_batch_data[2], dream_batch_data[2]], dim=0)
            next_states = torch.cat([real_batch_data[3], dream_batch_data[3]], dim=0)
            dones = torch.cat([real_batch_data[4], dream_batch_data[4]], dim=0)
            histories = torch.cat([real_batch_data[5], dream_batch_data[5]], dim=0)
            next_histories = torch.cat([real_batch_data[6], dream_batch_data[6]], dim=0)
            is_weights = torch.cat([real_batch_data[7], dream_batch_data[7]], dim=0)
        else:
            states, actions, rewards, next_states, dones, histories, next_histories, is_weights = real_batch_data
        
        # Текущие Q-значения
        current_q, gate_values = self.policy_net(states, histories)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        # Target Q-значения с Double DQN
        with torch.no_grad():
            # Выбор действий через policy net
            next_q_policy, _ = self.policy_net(next_states, next_histories)
            next_actions = next_q_policy.argmax(dim=1)
            
            # Оценка через target net
            next_q_target, _ = self.target_net(next_states, next_histories)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1))
            
            # N-step target
            target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * (GAMMA ** self.n_step) * next_q
        
        # TD ошибки
        td_errors = target_q - current_q
        
        # Huber loss с importance sampling
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        weighted_loss = (is_weights.unsqueeze(1) * elementwise_loss).mean()
        
        # Регуляризация gate
        gate_reg = torch.mean((gate_values - 0.5) ** 2) * 0.01
        total_loss = weighted_loss + gate_reg
        
        # Оптимизация
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Обновление приоритетов
        if dream_batch_size > 0:
            # Обновляем приоритеты для реальных данных
            self.update_priorities(real_indices, td_errors[:real_batch_size].squeeze(), use_dreams=False)
            # Обновляем приоритеты для снов
            self.update_priorities(dream_indices, td_errors[real_batch_size:].squeeze(), use_dreams=True)
        else:
            # Обновляем только реальные данные
            self.update_priorities(real_indices, td_errors.squeeze(), use_dreams=False)
        
        # Сохраняем loss
        self.training_losses.append(total_loss.item())
        self.target_update_counter += 1
        
        # Обновление target network
        if self.target_update_counter % 100 == 0:
            self.update_target()
        
        return total_loss.item()
    
    def update_target(self):
        """Мягкое обновление target network"""
        tau = 0.01
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
    
    def update_epsilon(self):
        """Обновление epsilon для exploration"""
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
    
    def generate_dreams(self, frozen_world_model, num_dreams=5, dream_length=10):
        """Генерация снов с использованием замороженной модели мира"""
        if not self.use_dreams or len(self.memory) < BATCH_SIZE:
            return
        
        for _ in range(num_dreams):
            # Выбираем случайное начальное состояние из реальной памяти
            idx = random.randint(0, len(self.memory) - 1)
            start_transition = self.memory[idx]
            
            # Создаем ImpressionBuffer для сна
            dream_buffer = ImpressionBuffer(seq_len=SEQ_LENGTH, states=STATES)
            
            # Восстанавливаем историю из перехода
            start_history = start_transition['history'].cpu().numpy()
            history_states = start_history.reshape(SEQ_LENGTH, STATES)
            for state in history_states:
                dream_buffer.add(state.tolist())
            
            # Генерируем сон
            for _ in range(dream_length):
                current_state = dream_buffer.get_sequence()
                history = dream_buffer.get_history_for_gate()
                
                # Выбираем действие через политику (без epsilon-greedy)
                with torch.no_grad():
                    state_tensor = torch.tensor(current_state, device=device, dtype=torch.float32).unsqueeze(0)
                    history_tensor = torch.tensor(history, device=device, dtype=torch.float32).unsqueeze(0)
                    q_values, _ = self.policy_net(state_tensor, history_tensor)
                    action = q_values.argmax().item()
                
                # Подготавливаем вход для модели мира
                action_one_hot = np.zeros(4, dtype=np.float32)
                action_one_hot[action] = 1.0
                
                world_model_input = np.concatenate([
                    current_state,          # 7
                    action_one_hot,         # 4
                    history                 # SEQ_LENGTH * 7 = 28
                ])
                
                # Предсказываем следующее состояние, награду и done
                with torch.no_grad():
                    world_model_input_tensor = torch.tensor(world_model_input, device=device, dtype=torch.float32).unsqueeze(0)
                    prediction = frozen_world_model.model(world_model_input_tensor)
                    prediction = prediction.cpu().numpy().flatten()
                
                # Разбираем предсказание
                next_state = prediction[:7]
                normalized_reward = prediction[7]
                done_prob = prediction[8]
                
                # Денормализуем награду
                reward = normalized_reward * 600.0
                done = done_prob > 0.5  # Порог для done
                
                # Создаем следующую историю
                next_history_buffer = ImpressionBuffer(seq_len=SEQ_LENGTH, states=STATES)
                for state_item in list(dream_buffer.buffer)[1:]:  # Берем последние SEQ_LENGTH-1 состояний
                    next_history_buffer.add(state_item)
                next_history_buffer.add(next_state.tolist())
                next_history = next_history_buffer.get_history_for_gate()
                
                # Сохраняем переход в память снов
                self.store_transition(
                    current_state,
                    action,
                    float(reward),
                    next_state.tolist(),
                    done,
                    history,
                    next_history,
                    is_dream=True
                )
                
                # Обновляем буфер сна
                dream_buffer.add(next_state.tolist())
                
                if done:
                    break

# Инициализация среды и агента
env = NumberGame()
agent = Agent(state_size=STATES, action_size=len(env.STRATEGIES))

# Инициализация World Model
world_model_input_size = 7 + 4 + SEQ_LENGTH * 7
world_model_output_size = 7 + 1 + 1  # state + reward + done
world_model = WorldModel(maxlen=10000, batch_size=256, 
                         input_size=world_model_input_size, 
                         hidden_size=128, 
                         output_size=world_model_output_size, lr=0.001)

# Замороженная модель мира (для генерации снов)
frozen_world_model = None

# Трекинг
reward_massive = []
record = 320
best_avg_reward = -float('inf')
losses = []
alpha_values = []
beta_values = []
gate_stats = []
world_model_losses = []
dream_usage_stats = []  # Статистика использования снов

# Параметры для снов
DREAM_START_EPISODE = 800  # Начинаем использовать сны после 1000 эпизодов
DREAM_BATCH_RATIO = 0.4  # 30% батча из снов
DREAMS_PER_STEP = 5  # Количество снов генерировать за шаг
DREAM_LENGTH = 50  # Длина одного сна

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    buffer = ImpressionBuffer(seq_len=SEQ_LENGTH, states=STATES)
    
    # Динамические параметры PER
    if episode < 2000:
        alpha = 0.5 + (episode / 2000) * 0.2
    else:
        alpha = 0.7 - min((episode - 2000) / 20000, 0.3)
    
    beta = min(1.0, 0.4 + (episode / 7000))
    
    # Сохраняем значения
    alpha_values.append(alpha)
    beta_values.append(beta)
    
    # Включаем использование снов после 1000 эпизодов
    if episode >= DREAM_START_EPISODE and not agent.use_dreams:
        agent.use_dreams = True
        # Создаем замороженную копию модели мира
        frozen_world_model = copy.deepcopy(world_model)
        frozen_world_model.model.eval()
        print(f"🎭 Включена генерация снов начиная с эпизода {episode}")
    
    if LOG_DETAILED and LOG_EPISODES_REMAINING > 0:
        print(f"\n🎯 Episode {episode} detailed log")
        print("Step | Action | HP | Mana | Enemy | Resist | Reward | Gate")
        print("-" * 70)
    
    step_count = 0
    episode_losses = []
    episode_gates = []
    episode_world_model_losses = []
    dream_steps = 0  # Счетчик шагов в снах за эпизод
 
    while not done:
        buffer.add(state)
        current_state = buffer.get_sequence()
        history = buffer.get_history_for_gate()
        
        # Выбор действия
        action, gate_vals = agent.act(current_state, history)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # История для следующего состояния
        next_history = buffer.get_history_for_gate()
        
        # Сохраняем transition в память агента (реальные данные)
        agent.store_transition(
            current_state, 
            action, 
            reward, 
            next_state, 
            done,
            history,
            next_history,
            is_dream=False
        )
        
        # Подготовка данных для World Model
        normalized_reward = np.clip(reward / 600.0, -1.0, 1.0)
        action_one_hot = np.zeros(4, dtype=np.float32)
        action_one_hot[action] = 1.0
        
        world_model_input = np.concatenate([
            current_state,
            action_one_hot,
            history
        ])
        
        world_model_target = np.concatenate([
            next_state,
            [normalized_reward],
            [float(done)]
        ])
        
        # Сохраняем в буфер World Model
        world_model.buffer.add(world_model_input, world_model_target)
        
        # Обучаем World Model
        if step_count % 4 == 0 and len(world_model.buffer) > world_model.buffer_batch_size:
            world_model_loss = world_model.train(num_epochs=1)
            if world_model_loss != -1:
                episode_world_model_losses.append(world_model_loss)
        
        # Генерация снов (если включена)
        if agent.use_dreams and frozen_world_model is not None:
            if step_count % 2 == 0:  # Генерируем сны каждые 5 шагов
                agent.generate_dreams(frozen_world_model, num_dreams=DREAMS_PER_STEP, dream_length=DREAM_LENGTH)
                dream_steps += DREAMS_PER_STEP * DREAM_LENGTH
        
        state = next_state
        
        # Записываем gate значения
        episode_gates.append(np.mean(gate_vals))
        if LOG_DETAILED and LOG_EPISODES_REMAINING > 0:
            action_names = ["⛑️⛑️", "⛑️✨", "✨✨", "⚔️✨"]
            event = "🏆 Win" if env.progress >= 65 and env.max_hp > 0 else "💀 Death" if env.max_hp <= 0 else ""
            print(f"{step_count:4} | {action_names[action]:<6} | {env.max_hp:5.1f} | {env.mana:5.1f} | {env.enemy_xp:5.1f} | {env.resist:6} | {reward:7.1f} | {event}")
        
        step_count += 1
        
        # Обучение агента с использованием снов
        if agent.use_dreams and len(agent.dream_memory) > 0:
            # Определяем размер батча для снов
            dream_batch_size = int(BATCH_SIZE * DREAM_BATCH_RATIO)
            loss = agent.learn(dream_batch_size=dream_batch_size)
        else:
            # Обучение только на реальных данных
            loss = agent.learn(dream_batch_size=0)
        
        if loss > 0:
            episode_losses.append(loss)
            
        if done:
            break
    
    reward_massive.append(total_reward)
    if episode_losses:
        losses.append(np.mean(episode_losses))
    
    # Сохраняем лосс World Model
    if episode_world_model_losses:
        avg_world_model_loss = np.mean(episode_world_model_losses)
        world_model_losses.append(avg_world_model_loss)
    
    # Сохраняем статистику использования снов
    if agent.use_dreams:
        dream_usage = len(agent.dream_memory) / (len(agent.memory) + len(agent.dream_memory) + 1e-6)
        dream_usage_stats.append(dream_usage)
    
    if LOG_DETAILED and LOG_EPISODES_REMAINING > 0:
        LOG_EPISODES_REMAINING -= 1
        if LOG_EPISODES_REMAINING == 0:
            LOG_DETAILED = False
            print("🔚 Detailed logging done")
    
    agent.update_epsilon()
    
    if episode % 1 == 0:
        release_params()
    
    if episode % 125 == 0:
        if len(reward_massive) > 0:
            recent_rewards = reward_massive[-100:] if len(reward_massive) >= 100 else reward_massive
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
        else:
            mean_reward = 0
            std_reward = 0
        
        avg_loss = np.mean(losses[-50:]) if len(losses) > 0 else 0
        
        # Добавляем лосс World Model в вывод
        if world_model_losses:
            avg_world_model_loss = np.mean(world_model_losses[-50:]) if len(world_model_losses) >= 50 else np.mean(world_model_losses)
        else:
            avg_world_model_loss = 0
        
        # Добавляем статистику снов
        dream_info = ""
        if agent.use_dreams:
            dream_ratio = len(agent.dream_memory) / (len(agent.memory) + 1e-6)
            dream_info = f", Dreams: {len(agent.dream_memory)} (ratio: {dream_ratio:.2f})"
        
        print(f"Episode {episode}, Epsilon: {agent.epsilon:.4f}, "
              f"Avg Reward: {mean_reward:.1f} ± {std_reward:.1f}, "
              f"Avg Loss: {avg_loss:.4f}, "
              f"World Model Loss: {avg_world_model_loss:.4f}"
              f"{dream_info}")
        
        # Обновление learning rate
        agent.scheduler.step(mean_reward)
        
        if mean_reward > best_avg_reward and mean_reward > record:
            best_avg_reward = mean_reward
            record = mean_reward
            print(f"🎉 New best average reward: {mean_reward:.1f}!")

            LOG_DETAILED = True
            LOG_EPISODES_REMAINING = 3
            print(f"Detailed logging enabled for next {LOG_EPISODES_REMAINING} episodes")