import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# 重新定義網路結構 (必須與訓練時一致)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def visualize():
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 建立環境 (重點：render_mode='human')
    env = gym.make("CartPole-v1", render_mode="human")
    
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 載入模型
    policy_net = DQN(n_observations, n_actions).to(device)
    try:
        policy_net.load_state_dict(torch.load("cartpole_model.pth"))
        policy_net.eval() # 設定為評估模式
        print("成功載入模型！")
    except FileNotFoundError:
        print("找不到模型檔案 'cartpole_model.pth'，請先執行 dqn_cartpole.py 進行訓練。")
        return

    num_episodes = 5 # 觀看幾場
    
    for i in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        done = False
        
        while not done:
            # 選擇動作 (不使用 Epsilon-Greedy，直接選最強的)
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            
            done = terminated or truncated
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            # 稍微暫停一下，讓畫面不要跑太快 (可選)
            # time.sleep(0.01)

        print(f"Episode {i+1}: Score = {total_reward}")

    env.close()

if __name__ == "__main__":
    visualize()
