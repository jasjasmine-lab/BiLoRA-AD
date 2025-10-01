# 傅立叶扩展模块使用说明

## 概述

`FourierExpansionModule` 是一个基于正交变换的神经网络模块，实现了 F-B-T 三层结构的傅立叶域变换。该模块使用严格的正交矩阵设计，确保变换的数学严谨性和计算效率。

## 核心特性

- **严格正交性**：使用 DCT-II 正交矩阵确保 F 和 T 层的正交性
- **方阵设计**：所有矩阵维度统一，简化计算逻辑
- **参数控制**：支持通过掩码精确控制 B 层参数更新
- **高效计算**：F 和 T 层冻结，只有 B 层参与训练

## 模块结构

```
输入 → T层(空域→频域) → B层(频域变换) → F层(频域→空域) → 输出
```

- **T层**：输入空域到频域的正交变换（冻结参数）
- **B层**：频域内的可训练线性变换
- **F层**：频域到输出空域的正交变换（冻结参数）

## API 参考

### 类初始化

```python
FourierExpansionModule(input_dim, mask=None)
```

**参数：**
- `input_dim` (int): 输入和输出维度，决定所有矩阵的大小
- `mask` (torch.Tensor, optional): B层的掩码矩阵，形状为 `(input_dim, input_dim)`
  - 值为1的位置：参数可以更新
  - 值为0的位置：参数被冻结，不会更新

### 主要方法

#### `forward(x)`
前向传播方法

**参数：**
- `x` (torch.Tensor): 输入张量，形状为 `[..., input_dim]`

**返回：**
- `torch.Tensor`: 输出张量，形状为 `[..., input_dim]`

#### `freeze_transform_matrices()`
冻结 F 和 T 层参数（默认已冻结）

#### `get_trainable_parameters()`
获取可训练参数列表

**返回：**
- `list`: 包含 B 层权重的参数列表

#### `get_frequency_weights()`
获取频域权重矩阵

**返回：**
- `torch.Tensor`: B 层的权重矩阵

#### `get_transform_matrices()`
获取变换矩阵

**返回：**
- `tuple`: (F矩阵的转置, T矩阵的转置)

## 使用示例

### 基本使用

```python
import torch
from fourier_expansion import FourierExpansionModule

# 创建模块
module = FourierExpansionModule(input_dim=64)

# 输入数据
x = torch.randn(10, 64)  # batch_size=10, input_dim=64

# 前向传播
output = module(x)
print(f"输入形状: {x.shape}")      # torch.Size([10, 64])
print(f"输出形状: {output.shape}")  # torch.Size([10, 64])
```

### 使用掩码控制参数更新

```python
# 创建掩码：只允许前32x32的参数更新
mask = torch.zeros(64, 64)
mask[:32, :32] = 1.0

# 创建带掩码的模块
masked_module = FourierExpansionModule(input_dim=64, mask=mask)

# 训练过程中，被掩码的参数不会更新
optimizer = torch.optim.Adam(masked_module.parameters())

for epoch in range(100):
    output = masked_module(x)
    loss = some_loss_function(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # 只有未被掩码的参数会更新
```

### 获取模块信息

```python
# 获取可训练参数
trainable_params = module.get_trainable_parameters()
print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")

# 获取频域权重
B_weights = module.get_frequency_weights()
print(f"B层权重形状: {B_weights.shape}")

# 获取变换矩阵
F_T, T_T = module.get_transform_matrices()
print(f"F矩阵转置形状: {F_T.shape}")
print(f"T矩阵转置形状: {T_T.shape}")
```

### 验证正交性

```python
# 验证T矩阵的正交性
T_matrix = module.T_layer.weight.data
orthogonality_check = torch.mm(T_matrix.t(), T_matrix)
identity_error = torch.norm(orthogonality_check - torch.eye(64))
print(f"T矩阵正交性误差: {identity_error.item():.6f}")

# 验证F=T^T关系
F_matrix = module.F_layer.weight.data
ft_relation_error = torch.norm(F_matrix - T_matrix.t())
print(f"F=T^T关系误差: {ft_relation_error.item():.6f}")
```

## 数学原理

### 正交变换
模块使用 DCT-II（离散余弦变换）作为正交基：

```
C[i,j] = sqrt(2/N) * cos(π * i * (2*j + 1) / (2*N))
```

其中 C[0,:] 行乘以额外的 1/sqrt(2) 因子以确保正交性。

### 变换关系
- T = C^T（DCT矩阵的转置）
- F = C（DCT矩阵）
- 满足：F = T^T，T^T * T = I

### 前向传播
```
y = F * B * T * x
```

其中：
- x: 输入向量
- T: 空域到频域变换
- B: 频域内的可训练变换
- F: 频域到空域变换
- y: 输出向量

## 注意事项

### 掩码机制
- 掩码通过梯度钩子实现，只影响参数更新，不影响前向传播
- 被掩码的参数在前向传播中仍然参与计算
- 如需在前向传播中也屏蔽某些连接，需要额外实现

### 内存和计算
- 所有矩阵都是方阵，内存使用为 O(input_dim²)
- F 和 T 层参数冻结，不参与梯度计算
- 只有 B 层参数需要存储梯度

### 数值稳定性
- DCT 变换具有良好的数值稳定性
- 正交性保证了变换的可逆性
- 建议使用适当的学习率避免 B 层参数过大

## 应用场景

1. **频域特征学习**：在频域内学习数据的表示
2. **结构化稀疏**：通过掩码实现结构化的参数稀疏
3. **预训练微调**：冻结部分频域参数，只微调特定部分
4. **信号处理**：利用傅立叶变换的性质处理信号数据

## 扩展建议

1. **多尺度变换**：支持不同尺度的傅立叶变换
2. **自适应掩码**：根据训练过程动态调整掩码
3. **其他正交基**：支持 DFT、Walsh 等其他正交变换
4. **批量归一化**：在 B 层后添加批量归一化提高稳定性

## 故障排除

### 常见问题

**Q: 正交性验证失败**
A: 检查 input_dim 是否正确，确保 DCT 矩阵计算无误

**Q: 掩码不起作用**
A: 确认掩码形状为 (input_dim, input_dim)，且值为 0 或 1

**Q: 训练不收敛**
A: 检查学习率设置，考虑对 B 层使用较小的学习率

**Q: 内存不足**
A: 减小 input_dim 或使用梯度检查点技术

---

更多详细信息请参考源代码中的注释和测试用例。