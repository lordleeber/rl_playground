
import torch
import time

def check_gpu():
    print("--- GPU Check ---")

    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(torch.version.cuda)               # 應顯示 12.8
        print(torch.backends.cudnn.version())   # 顯示 cuDNN 版本
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. Using CPU.")
        
    print("-----------------")


if __name__ == "__main__":
    check_gpu()
