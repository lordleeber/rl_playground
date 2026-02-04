import gymnasium as gym
import pygame
import sys
import time

def play():
    # 建立環境
    env = gym.make("CartPole-v1", render_mode="human")
    state, info = env.reset()
    
    print("遊戲開始！")
    print("控制方式：")
    print("  <- (左方向鍵): 向左推")
    print("  -> (右方向鍵): 向右推")
    print("  ESC: 退出遊戲")

    done = False
    total_reward = 0
    
    # 取得 pygame 視窗物件來監聽按鍵
    # 當 render_mode="human" 時，gymnasium 會初始化 pygame
    
    clock = pygame.time.Clock()

    while not done:
        action = None
        
        # 處理 pygame 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        # 檢查按鍵狀態 (持續按著也有效，但 CartPole 每步只能選一個動作)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1
        else:
            # 如果沒按鍵，預設隨機動作或不動作 (這裡我們選隨機，或者你可以設預設值)
            # 在 CartPole 中，每一幀都必須輸出一個動作 (0 或 1)
            action = env.action_space.sample() 

        # 執行動作
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        done = terminated or truncated
        
        # 控制遊戲速度 (CartPole 預設頻率很高)
        clock.tick(30) # 每秒 30 幀

    print(f"遊戲結束！你的總得分 (生存步數): {total_reward}")
    env.close()

if __name__ == "__main__":
    play()
