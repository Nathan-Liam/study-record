import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from train import ConfigParser, ResNet18, ResidualBlock

def visualize_predictions(config_path, checkpoint_path, num_images=9):
    # 初始化配置
    cfg = ConfigParser(config_path)
    device = torch.device(cfg.config['experiment']['device'] if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = ResNet18(
        ResidualBlock,
        cfg.config['model']['params']['num_blocks'],
        cfg.config['model']['params']['num_classes']
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 准备数据
    _, testloader = cfg.get_data_loaders()
    
    # 获取一批测试图像
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # 类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 显示原始图像
    images = images[:num_images]
    labels = labels[:num_images]
    
    # 反标准化
    mean = torch.tensor(cfg.config['data']['test_transform']['Normalize']['mean']).view(3, 1, 1)
    std = torch.tensor(cfg.config['data']['test_transform']['Normalize']['std']).view(3, 1, 1)
    unnormalized = images * std + mean
    unnormalized = torch.clamp(unnormalized, 0, 1)
    
    # 创建网格显示
    grid = make_grid(unnormalized, nrow=3)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    # 添加标题
    titles = []
    for i in range(len(images)):
        titles.append(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}')
    
    # 显示图像和预测结果
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(3, 3, i+1)
        plt.imshow(np.transpose(unnormalized[i].numpy(), (1, 2, 0)))
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(checkpoint_path), 'predictions.png'))
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--num_images', type=int, default=9, help='Number of images to visualize')
    args = parser.parse_args()
    
    visualize_predictions(args.config, args.checkpoint, args.num_images)