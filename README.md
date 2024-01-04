# 风格转换项目

这个项目实现了图像的风格转换功能，它包括两个主要部分：使用 TensorFlow Hub 的预训练模型进行风格迁移和使用 VGG19 模型的自定义风格迁移实现。

## 功能

- **HubModel**：使用 TensorFlow Hub 上的预训练模型进行风格迁移。
- **SelfModel**：使用 VGG19 模型的自定义实现进行风格迁移。

## 环境需求

- Python 3.x
- TensorFlow 2.x
- TensorFlow Hub
- Matplotlib
- Numpy

## 安装指南

确保您已经安装了上述所需库。如果没有安装，可以通过以下命令安装：

```bash
pip install tensorflow tensorflow-hub matplotlib numpy
