# 使用神经网络进行图像风格迁移的实现

每个实现之间是独立的，目前已经实现的（按实现时间顺序）：

- `cyclegan/` 即 CycleGAN，使用论文 [Unpaired Image-to-Image Translation
  using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- `cnn-transfer/` 使用论文 [*Image Style Transfer Using Convolutional Neural Networks*](https://ieeexplore.ieee.org/document/7780634)

## 说明

目录

- `assets/` 放置数据集
- `models/` 训练好的模型
- `output/` 风格迁移生成的图像
- `log/` 日志

文件

- `main.py` 使用其他脚本，模型训练和图像生成

  结构一般为

  1. 解析配置，得到 `opt`、`logger`；
  2. 构建模型、数据集、优化器，这部分会包装为函数再使用，防止污染外部命名空间；
  3. 图像生成函数的实现；
  4. 训练和生成函数的实现。

- `models.py` 模型结构实现

- `datasets.py` 数据集加载和预处理

- `utils.py` 日志记录和生成目录

一般来说，`main.py` 以外的文件是独立的。

## 使用

在每个实现的目录下都有配置文件 `options.yaml` 或配置目录 `configs/`。这些配置一般包含：项目命名，epoch 设置，模型和数据的保存，模型参数和优化器参数，图像大小（将会被裁剪或变形）。

然后按照配置将数据集放在 `assets/` 对应的子目录下，最后直接使用 `main.py` 脚本进行训练

~~~python
python main.py
~~~

