import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


plt.ion()

# 如果有 GPU 就用 GPU，否則用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定義一個簡單的經驗回放緩衝區 (Replay Buffer)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """儲存一個 transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """隨機取樣一個 batch"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 定義神經網路 (Q-Network)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # 簡單的三層全連接層網路
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 超參數設定
BATCH_SIZE = 128        # 每次訓練取樣的樣本數
GAMMA = 0.99            # 折扣因子 (Discount factor)
EPS_START = 0.9         # 探索率 (Epsilon) 初始值
EPS_END = 0.05          # 探索率 最終值
EPS_DECAY = 1000        # 探索率 衰減速度
TAU = 0.005             # 目標網路更新率 (Soft update)
LR = 1e-4               # 學習率

# 建立環境
env = gym.make("CartPole-v1")

# 取得狀態和動作的維度
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# 建立兩個網路：Policy Network (訓練用) 和 Target Network (目標用，穩定訓練)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    # 計算目前的探索率
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    # Epsilon-Greedy 策略
    if sample > eps_threshold:
        with torch.no_grad():
            # 選擇 Q 值最大的動作
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # 隨機探索
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 過濾掉最終狀態 (None)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 計算目前的 Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 計算下一時刻的最大 Q 值 V(s_{t+1})
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # 計算預期的 Q 值 = reward + gamma * next_q
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 計算 Loss (Huber Loss)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 優化模型
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪 (避免梯度爆炸)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# --- 主訓練迴圈 ---
num_episodes = 600 # 訓練回合數
print("開始訓練...")

episode_durations = []
best_avg_duration = 0  # 紀錄史上最高平均分數

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 儲存經驗
        memory.push(state, action, next_state, reward)
        state = next_state

        # 執行一步優化
        optimize_model()

        # 軟更新目標網路
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            
            # 計算最近 100 回合的平均值
            if len(episode_durations) >= 100:
                current_avg = sum(episode_durations[-100:]) / 100
            else:
                current_avg = sum(episode_durations) / len(episode_durations)

            # 如果平均表現創新高，就儲存模型
            if current_avg > best_avg_duration:
                best_avg_duration = current_avg
                torch.save(policy_net.state_dict(), "cartpole_model_best.pth")
                # print(f"New best average: {best_avg_duration:.2f} - Model saved!")

            if (i_episode + 1) % 10 == 0:
                print(f"Episode {i_episode + 1}/{num_episodes} | Current Avg: {current_avg:.1f} | Best Avg: {best_avg_duration:.1f}")
            break

print("訓練完成！")
env.close()

# 儲存最終模型 (目前的版本)
torch.save(policy_net.state_dict(), "cartpole_model_final.pth")
print(f"史上最高平均分數: {best_avg_duration:.2f}")
print("最優模型已儲存至 cartpole_model_best.pth")
print("最終模型已儲存至 cartpole_model_final.pth")

# 繪製並儲存訓練結果圖表
plt.figure(figsize=(10, 5))
plt.title("Training Result")
plt.xlabel("Episode")
plt.ylabel("Duration (Steps)")
plt.plot(episode_durations, label='Episode Duration')

# 繪製 100 回合的移動平均線 (更能看出趨勢)
if len(episode_durations) >= 100:
    means = torch.tensor(episode_durations, dtype=torch.float).unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy(), label='100-episode Moving Average')

plt.legend()
plt.savefig("training_result.png")
print("訓練結果圖表已儲存至 training_result.png")

# 簡單的視覺化結果 (文字版)
print(f"平均生存步數 (最後 10 回合): {sum(episode_durations[-10:])/10:.1f}")
