# Audio BSQ-ViT (音频版本)

这是 BSQ-ViT 针对音频/语音处理的改编版本，参考了 AudioMAE 和 Conformer 等模型的架构设计。

## 概述

音频版 BSQ-ViT 使用基于 Transformer 的架构处理音频梅尔频谱图(mel-spectrograms)，配备可选的 Conformer 风格卷积模块以增强局部特征提取能力。

## 主要特性

- **音频 Transformer 架构**: 从视觉 Transformer 改编用于处理音频频谱图
- **Conformer 风格卷积模块**: 可选的卷积模块用于增强局部特征建模
- **二值球面量化(BSQ)**: 使用二值量化实现高效的音频标记化
- **灵活的分块处理**: 可配置的时间和频率分块大小
- **多数据集支持**: LibriSpeech、Common Voice 和通用音频文件夹

## 快速开始

### 安装

```bash
# 创建环境
mamba env create -f bsqvit-env.yaml
mamba activate bsqvit

# 安装音频依赖
mamba install -c conda-forge torchaudio librosa
```

### 测试模型

```bash
python scripts/test_audio_model.py
```

### 准备数据

#### 选项 1: 使用 LibriSpeech

```bash
# 下载 LibriSpeech 训练集
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# 下载验证集
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
```

#### 选项 2: 使用自己的音频数据

```bash
# 组织音频文件结构:
mkdir -p my_audio/train
mkdir -p my_audio/val

# 将训练音频放在 train/ 目录
# 将验证音频放在 val/ 目录
```

### 开始训练

```bash
# 单 GPU 训练
python scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_simple_bsq_b18.yaml \
    --output-dir ./outputs/audio_model

# 多 GPU 训练
torchrun --nproc_per_node=4 scripts/main_audio_tokenizer.py \
    configs/tokenizer/audio_librispeech_bsq_b18.yaml \
    --output-dir ./outputs/audio_model
```

## 架构组件

### AudioTransformerEncoder (音频 Transformer 编码器)
- 使用分块标记化处理梅尔频谱图
- 支持 Conformer 风格卷积模块用于局部特征提取
- 为时间-频率分块提供可学习的位置嵌入

### AudioTransformerDecoder (音频 Transformer 解码器)
- 从量化表示重建频谱图
- 与编码器镜像的架构

### AudioBSQModel (音频 BSQ 模型)
- 完整的音频标记化模型
- 使用二值球面量化实现高效的离散表示
- 可配置的量化参数

## 配置说明

主要配置参数:

### 模型参数
- `embed_dim`: 量化嵌入维度 (例如 18 表示 18 位编码)
- `input_size`: 梅尔频率通道数 (默认: 128)
- `input_length`: 时间帧数 (默认: 1024)
- `patch_size`: 时间和频率维度的分块大小
- `use_conv_module`: 启用 Conformer 风格卷积 (推荐: True)
- `conv_kernel_size`: 卷积模块的核大小 (默认: 31)

### 音频处理参数
- `sample_rate`: 音频采样率 (默认: 16000 Hz)
- `n_mels`: 梅尔频率通道数 (默认: 128)
- `n_fft`: FFT 大小 (默认: 1024)
- `hop_length`: STFT 的跳跃长度 (默认: 160)
- `target_length`: 目标帧长度 (默认: 1024)

## 与图像版本的主要区别

| 方面 | 图像 BSQ-ViT | 音频 BSQ-ViT |
|------|-------------|-------------|
| 输入格式 | RGB 图像 | 梅尔频谱图 |
| 输入形状 | (B, 3, 256, 256) | (B, 1024, 128) |
| 分块策略 | 空间分块 | 时间-频率分块 |
| 卷积模块 | 无 | Conformer 风格 |
| 归一化 | [0,1] 或 [-1,1] | [-80,0] dB |
| 感知损失 | LPIPS (VGG) | 无 |
| 数据增强 | 裁剪、翻转 | SpecAugment |

## 设计参考

### AudioMAE 启发
1. 基于分块的频谱图处理
2. 时间-频率位置嵌入
3. 掩码自编码能力 (可添加)
4. 频谱图输入而非原始波形

### Conformer 启发
1. 用于局部特征提取的卷积模块
2. 卷积层中的 GLU 激活
3. 卷积路径中的批归一化
4. 结合全局注意力和局部建模

## 模型变体

### 标准配置 (18位)
- `embed_dim: 18`
- `width: 768`
- `layers: 12`
- `heads: 12`

### 大型配置 (36位)
- `embed_dim: 36`
- `width: 1024`
- `layers: 24`
- `heads: 16`

### 小型配置 (9位)
- `embed_dim: 9`
- `width: 512`
- `layers: 6`
- `heads: 8`

## 使用示例

```python
import torch
from transcoder.models.audiobsq import AudioBSQModel

# 创建模型
config = {
    'input_size': 128,
    'input_length': 1024,
    'patch_size': 16,
    'width': 768,
    'layers': 12,
    'heads': 12,
    'use_conv_module': True,
}

model = AudioBSQModel(
    audioconfig=config,
    embed_dim=18,
    l2_norm=True,
)

# 处理音频 (梅尔频谱图)
audio = torch.randn(4, 1024, 128)  # (batch, time, freq)
reconstructed, quant_loss, info = model(audio)

# 访问量化编码
encoded, _, _ = model.encode(audio)
print(f"编码后形状: {encoded.shape}")
```

## 评估指标

模型输出:
- **MSE**: 频谱图上的均方误差
- **MAE**: 频谱图上的平均绝对误差
- **重建损失**: 整体重建质量

## 常见问题

### 内存不足
- 减少 `batch_size`
- 减少 `input_length` (时间帧数)
- 启用梯度检查点: config 中设置 `grad_checkpointing: True`

### 重建质量差
- 增加模型容量 (`width`, `layers`)
- 增加 `embed_dim` (更多量化位)
- 启用 Conformer 卷积模块
- 调整 `clamp_range` 以匹配音频数据

### 训练不稳定
- 降低学习率
- 增加预热步数
- 检查音频归一化 (应为 dB 尺度)
- 启用梯度裁剪 (默认已启用)

## 适用场景

### 语音处理 (LibriSpeech, Common Voice)
```yaml
data:
  sample_rate: 16000
  n_mels: 80  # 语音使用较少的梅尔通道
  
model:
  params:
    audioconfig:
      use_conv_module: True  # 推荐用于语音
```

### 音乐处理
```yaml
data:
  sample_rate: 22050  # 更高采样率
  n_mels: 128
  n_fft: 2048
  target_length: 2048  # 更长上下文
```

### 环境声音
```yaml
data:
  sample_rate: 16000
  n_mels: 128
  target_length: 512  # 更短上下文
```

## 文档

详细文档请参考:
- [AUDIO_README.md](AUDIO_README.md) - 完整文档
- [AUDIO_ARCHITECTURE.md](AUDIO_ARCHITECTURE.md) - 架构细节
- [QUICKSTART_AUDIO.md](QUICKSTART_AUDIO.md) - 快速开始指南
- [CHANGELOG_AUDIO.md](CHANGELOG_AUDIO.md) - 更改日志

## 引用

如果使用此音频改编版本，请引用原始 BSQ-ViT 论文以及启发性工作:

```bibtex
@article{zhao2024bsqvit,
  title={Image and Video Tokenization with Binary Spherical Quantization},
  author={Zhao, Yue and Xiong, Yuanjun, and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2406.07548},
  year={2024}
}

@article{huang2022audiomae,
  title={Masked autoencoders that listen},
  author={Huang, Po-Yao and Xu, Hu and Li, Juncheng and others},
  journal={NeurIPS},
  year={2022}
}

@article{gulati2020conformer,
  title={Conformer: Convolution-augmented transformer for speech recognition},
  author={Gulati, Anmol and Qin, James and others},
  journal={INTERSPEECH},
  year={2020}
}
```

## 后续改进

未来可能的增强:
- [ ] 掩码自编码预训练 (AudioMAE 风格)
- [ ] 多尺度频谱图处理
- [ ] 声码器集成用于音频合成
- [ ] 更多音频数据集支持
- [ ] 音频专用评估指标 (SI-SNR, PESQ, STOI)
- [ ] 流式音频处理支持

## 许可证

与原始 BSQ-ViT 相同: MIT License

---

如有问题或建议，请在 GitHub 上提 issue。
