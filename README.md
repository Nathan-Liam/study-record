# CIFAR10 图像分类项目

## 项目结构

```text
CIFAR10/
├── configs/               
│   ├── baseline.yaml      
│   ├── exp001_lr0.1.yaml  
│   └── exp002_lr0.01.yaml 
├── scripts/               
│   ├── train.py           
│   └── visualize.py       
├── data/                  
└── runs/                  
```

configs里面放每个实验的配置文件

scripts里面放通用代码

runs里面存放每个实验的结果

改配置文件即可进行不同实验


## 训练模型

python scripts/train.py --config configs/baseline.yaml

--config 参数改成自己要的配置文件

运行后，会在runs生成相应的文件夹，文件夹名为配置文件里写的name+当前时间


## 可视化结果
可查看一组图片在某模型下的分类结果

python scripts/visualize.py --config configs/baseline.yaml --checkpoint runs/baseline_20231115/checkpoint.pth

--config 模型配置

--checkpoint 检查点的模型参数文件，参数文件较大没有上传


## 查看训练曲线
用tensorboard记录了训练过程

tensorboard --logdir=runs\baseline_resnet18_20250515_215425\logs

我完整运行了baseline模型，其余的都只是简单的训练尝试，证明代码不存在问题

测试集结果准确度是91.87%，比VGG好

![image](https://github.com/user-attachments/assets/d43eab0d-3f09-4af7-a3c0-6aab40e8fbee)

## 分阶段调参
分阶段调参时，参数之间存在关联性无法避免，因此分阶段调参会先调影响大的，再调影响小的

网格搜索/随机搜索，贝叶斯优化这些方法，通过不断的遍历去寻找最优组合，一般用在机器学习上，深度学习所需运算资源较大，用的少

谷歌搞出来一本调参手册，看了一点

https://github.com/google-research/tuning_playbook

第一步，先选一个通用的模型架构

第二步，选择一个优化器，然后确定batchsize，靠近所用计算资源的极限，越大越好

以上建立了一个完整的训练pipeline，和合适的初始配置，这就是baseline.yaml

接下来从简单的配置开始，设计一轮一轮的实验进行调参，每一轮实验设置一个目标，比如，确定哪个优化器在给定的步数中产生最低的验证错误，此时，优化器便是科学超参数

然后，其他超参一些是固定参数，比如batchsize，还有一些是可以变化的，叫干扰参数，比如学习率

在平衡性能的条件下，尽可能减少干扰超参数，把干扰超参数当成固定超参数，因为不可能去探索那么多。不过干扰超参数与科学超参数的交互作用越强，固定其值的破坏性就越大，所以依据经验决定。

然后，依据我们找到的最好的干扰超参数，然后比较科学超参数，这就完成了一轮实验调参

可以根据想要的目标设计下一轮实验调参，比如我要找到合适的模型层数，这个时候，优化器便当成了固定参数。然后以此类推。


