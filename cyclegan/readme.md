## 额外说明

### 配置

数据集不一点需要放在 `assets/` 中，可以在 `configs/cycle_options.yaml` 的 `root` 项中指定，但其中的训练集要命名为 `trainA` 和 `trainB`，测试集要命名为 `testA` 和 `testB`。

日志由 `configs/cycle_logger.yaml` 配置，默认输出到 `cycle_log.log` 中。

### 训练注意

使用 Adam 优化器，而且 b2 参数比较高。

学习率将在 `decay_epoch` 后进行倒数衰减，即到达训练次数 `n_epochs` 后学习率几乎为 0。

每训练一次生成器 G 训练 `n_train_D` 次鉴别器 D，多次训练鉴别器可以促进生成器的进步。

CycleGAN 重要的两个损失 cycle loss 和 identity loss。

**可能效果不会太好**。