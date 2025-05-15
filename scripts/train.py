import os
import yaml
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class ConfigParser:
    def __init__(self, config_path, make_run_dir=True):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # 设置随机种子
        if 'seed' in self.config['experiment']:
            torch.manual_seed(self.config['experiment']['seed'])
        
        # 自动创建目录
        os.makedirs(self.config['data']['root'], exist_ok=True)

        if make_run_dir==True:
            self.run_dir = os.path.join(
                self.config['logging']['save_dir'],
                f"{self.config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(self.run_dir, exist_ok=True)
        
            # 保存配置副本
            with open(os.path.join(self.run_dir, 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)
        
            # 初始化TensorBoard
            self.writer = SummaryWriter(os.path.join(self.run_dir, 'logs'))

    def get_data_loaders(self):
        """创建数据加载器"""
        train_transform = self._build_transform(self.config['data']['train_transform'])
        test_transform = self._build_transform(self.config['data']['test_transform'])
        
        trainset = torchvision.datasets.CIFAR10(
            root=self.config['data']['root'],
            train=True,
            download=True,
            transform=train_transform
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=self.config['data']['root'],
            train=False,
            download=True,
            transform=test_transform
        )
        
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.config['data']['batch_size']['train'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config['data']['batch_size']['test'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        return trainloader, testloader

    def _build_transform(self, transform_config):
        """动态构建transform"""
        transform_list = []
        
        if 'RandomCrop' in transform_config:
            transform_list.append(
                transforms.RandomCrop(
                    transform_config['RandomCrop']['size'],
                    padding=transform_config['RandomCrop'].get('padding', 0)
                )
            )
        
        if 'RandomHorizontalFlip' in transform_config and transform_config['RandomHorizontalFlip']:
            transform_list.append(transforms.RandomHorizontalFlip())
            
        if 'RandomRotation' in transform_config:
            transform_list.append(transforms.RandomRotation(transform_config['RandomRotation']))
            
        if 'ColorJitter' in transform_config:
            transform_list.append(transforms.ColorJitter(
                brightness=transform_config['ColorJitter']['brightness'],
                contrast=transform_config['ColorJitter']['contrast'],
                saturation=transform_config['ColorJitter']['saturation'],
                hue=transform_config['ColorJitter']['hue']
            ))
            
        if 'RandomAffine' in transform_config:
            transform_list.append(transforms.RandomAffine(
                degrees=transform_config['RandomAffine']['degrees'],
                translate=transform_config['RandomAffine']['translate'],
                scale=transform_config['RandomAffine']['scale']
            ))
            
        transform_list.append(transforms.ToTensor())
        
        if 'Normalize' in transform_config:
            transform_list.append(transforms.Normalize(
                mean=transform_config['Normalize']['mean'],
                std=transform_config['Normalize']['std']
            ))
            
        if 'RandomErasing' in transform_config:
            transform_list.append(transforms.RandomErasing(
                p=transform_config['RandomErasing']['p'],
                scale=transform_config['RandomErasing']['scale']
            ))
            
        return transforms.Compose(transform_list)

    def get_model(self):
        """创建模型"""
        if self.config['model']['type'] == "ResNet18":
            return ResNet18(
                block=ResidualBlock,
                num_blocks=self.config['model']['params']['num_blocks'],
                num_classes=self.config['model']['params']['num_classes']
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model']['type']}")

    def get_optimizer(self, model):
        """创建优化器"""
        opt_config = self.config['training']['optimizer']
        if opt_config['type'] == "SGD":
            return optim.SGD(
                model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")

    def get_scheduler(self, optimizer):
        """创建学习率调度器"""
        sched_config = self.config['training']['lr_scheduler']
        if sched_config['type'] == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_config['T_max']
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_config['type']}")

    def get_criterion(self):
        """创建损失函数"""
        if self.config['training']['criterion'] == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown criterion: {self.config['training']['criterion']}")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu= nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.out_channel = 64
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.linear = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(block, num_blocks, num_classes=10):
    return ResNet(block, num_blocks, num_classes)

def train(model, trainloader, criterion, optimizer, device, epoch, writer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(trainloader)} | '
                  f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')
    
    avg_loss = train_loss/len(trainloader)
    avg_acc = 100.*correct/total
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_acc, epoch)
    
    return avg_loss, avg_acc

def test(model, testloader, criterion, device, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = test_loss/len(testloader)
    avg_acc = 100.*correct/total
    
    print(f'Test | Loss: {avg_loss:.3f} | Acc: {avg_acc:.3f}%')
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Accuracy/test', avg_acc, epoch)
    
    return avg_loss, avg_acc

def save_checkpoint(model, optimizer, scheduler, epoch, run_dir, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    
    filename = os.path.join(run_dir, f'checkpoint_epoch{epoch}.pth')
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(run_dir, 'model_best.pth')
        torch.save(state, best_filename)

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 初始化配置
    cfg = ConfigParser(args.config)
    device = torch.device(cfg.config['experiment']['device'] if torch.cuda.is_available() else "cpu")
    
    # 准备数据
    trainloader, testloader = cfg.get_data_loaders()
    
    # 初始化模型
    model = cfg.get_model().to(device)
    
    # 初始化优化器和损失函数
    criterion = cfg.get_criterion()
    optimizer = cfg.get_optimizer(model)
    scheduler = cfg.get_scheduler(optimizer)
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(1, cfg.config['training']['epochs'] + 1):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device, epoch, cfg.writer)
        test_loss, test_acc = test(model, testloader, criterion, device, epoch, cfg.writer)
        scheduler.step()
        
        # 保存checkpoint
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        
        if epoch % cfg.config['logging']['save_interval'] == 0 or epoch == cfg.config['training']['epochs']:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg.run_dir, is_best)
    
    
    # 保存指标
    metrics = {
        'best_accuracy': best_acc,
        'final_train_accuracy': train_acc,
        'final_test_accuracy': test_acc
    }
    with open(os.path.join(cfg.run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    cfg.writer.close()

if __name__ == '__main__':
    main()