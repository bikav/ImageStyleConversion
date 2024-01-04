# ImageStyleConversion
这个项目实现了图像的风格转换功能，它包括两个主要部分：使用 TensorFlow Hub 的预训练模型进行风格迁移和使用 VGG19 模型的自定义风格迁移实现。

功能
HubModel：使用 TensorFlow Hub 上的预训练模型进行风格迁移。
SelfModel：使用 VGG19 模型的自定义实现进行风格迁移。
环境需求
Python 3.x
TensorFlow 2.x
TensorFlow Hub
Matplotlib
Numpy
安装指南
确保您已经安装了上述所需库。如果没有安装，可以通过以下命令安装：

bash
Copy code
pip install tensorflow tensorflow-hub matplotlib numpy

使用说明
HubModel 使用：

创建 HubModel 实例。
使用 style_conver 方法进行风格迁移。此方法需要内容图像路径、风格图像路径和结果保存路径。
示例代码：

python
Copy code
hubModel = HubModel()
hubModel.style_conver('result_path', 'content_image_path', 'style_image_path')
SelfModel 使用：

创建 SelfModel 实例。
使用 style_conver 方法进行风格迁移。此方法需要内容图像路径、风格图像路径和结果保存路径。
示例代码：

python
Copy code
selfModel = SelfModel()
selfModel.style_conver('result_path', 'content_image_path', 'style_image_path')

文件结构
image/：存放原始图像和风格图像的文件夹。
result/：存放风格转换结果的文件夹。
