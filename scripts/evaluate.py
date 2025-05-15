import os
import torch
import json
from train import ConfigParser, ResNet18, ResidualBlock

def evaluate(config_path, checkpoint_path):
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
    
    # 评估
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # 保存结果
    result = {'test_accuracy': accuracy}
    with open(os.path.join(os.path.dirname(checkpoint_path), 'eval_results.json'), 'w') as f:
        json.dump(result, f)
    
    return accuracy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint)