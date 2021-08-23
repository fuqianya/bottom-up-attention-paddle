# bottom-up-attention-paddle

基于[paddle](https://github.com/PaddlePaddle/Paddle)框架的[Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)实现

## 一、简介

本项目基于[paddle](https://github.com/PaddlePaddle/Paddle)复现[Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)中所提出的基于`bottom-up`和`top-down`注意力机制的`Image Captioning`模型。论文作者提出了著名的`bottom-up`注意力机制，与以往的`grid-level`的注意力不同，作者提出了`object-level`的注意力机制。作者将该注意力机制应用到`image captioning`和`visual question answering (vqa)`任务中，均取得了显著的效果。

**论文:**

* [1] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, L. Zhang, "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering", CVPR, 2018.

**参考项目:**

* [https://github.com/peteanderson80/Up-Down-Captioner](https://github.com/peteanderson80/Up-Down-Captioner) [官方实现]

* [https://github.com/ezeli/BUTD_model](https://github.com/ezeli/BUTD_model)

## 二、复现精度

> 该指标为模型在[COCO2014](https://cocodataset.org/)的测试集评估而得

| 指标 | 原论文 | 复现精度 | 
| :---: | :---: | :---: | 
| BlEU-1 | 0.798 | 0.791 |

## 三、数据集

本项目所使用的数据集为[COCO2014](https://cocodataset.org/)。该数据集共包含123287张图像，每张图像对应5个标题。训练集、验证集和测试集分别为113287、5000、5000张图像及其对应的标题。本项目使用作者提供的预提取的`bottom-up`特征，可以从[这里](https://github.com/peteanderson80/bottom-up-attention)下载得到（我们提供了脚本下载该数据集的标题以及图像特征，见[download_dataset.sh](https://github.com/fuqianya/bottom-up-attention-paddle/blob/main/download_dataset.sh)）。

## 四、环境依赖

* 硬件：CPU、GPU

* 软件：
    * Python 3.8
    * Java 1.8.0
    * PaddlePaddle == 2.1.0

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/fuqianya/bottom-up-attention-paddle.git
cd bottom-up-attention-paddle
```

### step2: 安装环境及依赖

```bash
pip install -r requirements.txt
```

### step3: 下载数据

```bash
# 下载数据集及特征
bash ./download_dataset.sh
```

### step4: 数据集预处理

```python
python prepro.py
```

### step5: 训练

训练过程过程分为两步(详情见论文3.3节):

* Training with Cross Entropy (XE) Loss

  ```bash
  python train.py --train_mode xe --learning_rate 4e-4
  ```

* CIDEr-D Score Optimization

  ```bash
  python train.py --train_mode rl --learning_rate 4e-5 --resume ./checkpoint/xe/epoch_25.pth
  ```
### step6: 测试

```bash
python eval.py --train_mode rl --eval_model ./checkpoint/rl/epoch_25.pth --result_file epoch25_results.json
```

### 使用预训练模型进行预测

模型下载: [谷歌云盘](https://drive.google.com/drive/folders/1_ShwBUsUir33VHXtLSypn44Ah-Ke5jyZ?usp=sharing)

将下载的模型权重放到`checkpoints`目录下, 运行`step6`的指令进行测试。

## 六、代码结构与详细说明

```bash
├── checkpoint      　   # 存储训练的模型
├── config
│　 └── config.py        # 模型的参数设置
├── data            　   # 预处理的数据
├── model
│   └── captioner.py   　# 定义模型结构
│   └── dataloader.py  　# 加载训练数据
│   └── loss.py        　# 定义损失函数
├── pyutils 
│   └── cap_eval       　# 计算评价指标工具
│   └── self_critical  　# rl阶段计算reward工具
├── result            　 # 存放生成的标题
├── utils 
│   └── utils.py       　# 工具类
├── download_dataset.sh　# 数据集下载脚本
├── prepro.py          　# 数据预处理
├── train.py           　# 训练主函数
├── eval.py            　# 测试主函数
└── requirement.txt   　 # 依赖包
```

模型、训练的所有参数信息都在`config.py`中进行了详细注释，详情见`config/config.py`。

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| :---: | :---: |
| 发布者 | fuqianya |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.0 |
| 应用场景 | 多模态 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://drive.google.com/drive/folders/1_ShwBUsUir33VHXtLSypn44Ah-Ke5jyZ?usp=sharing) \| [训练日志](https://drive.google.com/drive/folders/1_ShwBUsUir33VHXtLSypn44Ah-Ke5jyZ?usp=sharing)  |