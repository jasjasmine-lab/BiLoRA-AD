import torch
import torch.nn as nn
import numpy as np


class MaskMatrix:
    """
    掩码矩阵管理类
    
    用于管理和分配掩码矩阵，维护矩阵的占用状态，确保新分配的掩码不会覆盖已占用的位置。
    
    功能特点：
    - 维护一个二进制状态矩阵（0表示空闲，1表示已占用）
    - 支持固定大小的掩码分配
    - 自动更新内部状态，避免位置冲突
    """
    
    def __init__(self, matrix_shape, allocation_size):
        """
        初始化掩码矩阵管理器
        
        Args:
            matrix_shape (tuple): 掩码矩阵的维度，格式为(m, n)
            allocation_size (int): 每次分配的固定数量（分配后不可更改）
        """
        self.matrix_shape = matrix_shape
        self.allocation_size = allocation_size
        
        # 初始化状态矩阵，0表示空闲，1表示已占用
        self.state_matrix = torch.zeros(matrix_shape, dtype=torch.int32)
        
        # 记录总的可用位置数
        self.total_positions = matrix_shape[0] * matrix_shape[1]
        self.allocated_positions = 0
        
    def get_available_positions(self):
        """
        获取当前可用位置的数量
        
        Returns:
            int: 可用位置数量
        """
        return self.total_positions - self.allocated_positions
    
    def get_state_matrix(self):
        """
        获取当前状态矩阵的副本
        
        Returns:
            torch.Tensor: 状态矩阵副本
        """
        return self.state_matrix.clone()
    
    def create_mask(self):
        """
        在空闲位置创建新的掩码
        
        在当前状态矩阵中找到空闲位置（值为0的位置），分配指定数量的位置，
        将这些位置标记为已占用（设为1），并返回新创建的掩码矩阵。
        
        Returns:
            torch.Tensor: 新创建的掩码矩阵，形状与matrix_shape相同
                         新分配的位置为1，其他位置为0
                         
        Raises:
            RuntimeError: 当可用位置不足时抛出异常
        """
        # 检查是否有足够的空闲位置
        available_count = self.get_available_positions()
        if available_count < self.allocation_size:
            raise RuntimeError(
                f"可用位置不足：需要 {self.allocation_size} 个位置，但只有 {available_count} 个可用位置"
            )
        
        # 找到所有空闲位置的索引
        free_positions = torch.where(self.state_matrix == 0)
        free_indices = list(zip(free_positions[0].tolist(), free_positions[1].tolist()))
        
        # 随机选择要分配的位置（也可以按顺序选择）
        import random
        selected_indices = random.sample(free_indices, self.allocation_size)
        
        # 创建新的掩码矩阵
        new_mask = torch.zeros(self.matrix_shape, dtype=torch.float32)
        
        # 更新状态矩阵和新掩码
        for row, col in selected_indices:
            self.state_matrix[row, col] = 1  # 标记为已占用
            new_mask[row, col] = 1.0  # 在新掩码中标记
        
        # 更新已分配位置计数
        self.allocated_positions += self.allocation_size
        
        return new_mask
    
    def reset(self):
        """
        重置状态矩阵，清空所有分配
        """
        self.state_matrix.fill_(0)
        self.allocated_positions = 0
    
    def get_allocation_info(self):
        """
        获取分配信息
        
        Returns:
            dict: 包含分配统计信息的字典
        """
        return {
            'matrix_shape': self.matrix_shape,
            'allocation_size': self.allocation_size,
            'total_positions': self.total_positions,
            'allocated_positions': self.allocated_positions,
            'available_positions': self.get_available_positions(),
            'utilization_rate': self.allocated_positions / self.total_positions
        }


class FourierExpansionModule(nn.Module):
    """
    傅立叶参数扩展组件
    
    实现三层结构：
    1. F矩阵乘法层（冻结）- 将频域转换到输出空域
    2. B矩阵线性层（可训练）- 在频域进行参数变换
    3. T矩阵乘法层（冻结）- 将输入空域转换到频域
    
    核心公式：output = F * B * T * input
    其中F和T是固定的正交DCT矩阵，只有B是可训练的
    
    设计特点：
    - 严格使用正交方阵：所有矩阵维度统一为input_dim
    - F和T矩阵复用同一个正交DCT矩阵（F=T^T）
    - 保证数学上的正交性：C^T * C = I
    """
    
    def __init__(self, input_dim, mask=None):
        """
        初始化傅立叶扩展模块
        
        Args:
            input_dim (int): 输入和输出维度，决定所有矩阵的维度
            mask (torch.Tensor, optional): B矩阵的掩码矩阵，形状为(input_dim, input_dim)
                - 如果为None，不使用掩码
                - 如果提供掩码，将应用到B层的权重上
        """
        super(FourierExpansionModule, self).__init__()
        
        self.input_dim = input_dim
        
        # 注册掩码矩阵
        if mask is not None:
            self.register_buffer('mask', mask)
        else:
            self.register_buffer('mask', torch.ones(input_dim, input_dim))
        
        # 创建正交DCT矩阵
        orthogonal_matrix = self._create_orthogonal_dct_matrix(input_dim)
        
        # 第一层：F矩阵线性层（冻结）- 频域到空域的变换
        # F = T^T（正交矩阵的转置）
        self.F_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.F_layer.weight.data = orthogonal_matrix.clone()  # F = T^T
        self.F_layer.weight.requires_grad = False  # 冻结F层
        
        # 第二层：B矩阵线性层（可训练）- 频域线性变换
        self.B_layer = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.eye_(self.B_layer.weight)  # 初始化为单位矩阵
        
        # 第三层：T矩阵线性层（冻结）- 输入空域到频域的变换
        # T矩阵使用正交DCT矩阵的转置
        self.T_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.T_layer.weight.data = orthogonal_matrix.t()  # T = C^T
        self.T_layer.weight.requires_grad = False  # 冻结T层
        # 如果提供了掩码，注册梯度钩子来阻止被掩码位置的参数更新
        if mask is not None:
            self.B_layer.weight.register_hook(self._mask_gradient_hook)
        
    
    def _mask_gradient_hook(self, grad):
        """
        梯度钩子函数，用于将被掩码位置的梯度置零
        
        Args:
            grad (torch.Tensor): B层权重的梯度
            
        Returns:
            torch.Tensor: 应用掩码后的梯度
        """
        if self.mask is not None:
            return grad * self.mask
        return grad
        
    def _create_orthogonal_dct_matrix(self, N):
        """
        创建正交的DCT-II变换矩阵
        
        Args:
            N (int): 矩阵维度
            
        Returns:
            torch.Tensor: 正交DCT矩阵 [N, N]，满足 C^T * C = I
        """
        # 创建空间和频率索引
        n = torch.arange(N, dtype=torch.float32).view(1, -1)  # [1, N] 空间索引
        k = torch.arange(N, dtype=torch.float32).view(-1, 1)  # [N, 1] 频率索引
        
        # DCT-II角度公式：π(2n+1)k/(2N)
        angles = np.pi * (2 * n + 1) * k / (2 * N)
        
        # 计算DCT-II矩阵元素
        dct_matrix = torch.cos(angles)
        
        # 归一化因子
        # 对于k=0（第一行），归一化因子是sqrt(1/N)
        # 对于k>0，归一化因子是sqrt(2/N)
        normalization = torch.ones(N, 1) * np.sqrt(2.0 / N)
        normalization[0, 0] = np.sqrt(1.0 / N)  # 第一行特殊处理
        
        # 应用归一化
        dct_matrix = dct_matrix * normalization
        
        return dct_matrix
    
    def forward(self, x):
        """
        前向传播：实现 output = F(B(T(x))) 的三层结构，并应用掩码
        
        Args:
            x (torch.Tensor): 输入张量 [..., input_dim]
            
        Returns:
            torch.Tensor: 输出张量 [..., input_dim]
            
        计算流程：
        1. T层：将输入从空域变换到频域 (input_dim -> input_dim)
        2. B层：在频域进行可训练的线性变换 (input_dim -> input_dim)，应用掩码
        3. F层：将频域结果变换到空域 (input_dim -> input_dim)
        """
        # 保存原始形状
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        
        # 重塑为二维进行线性层运算
        x_flat = x.view(-1, self.input_dim)  # [batch_size, input_dim]
        
        # 第三层：T线性层（输入空域到频域变换）
        x_freq = self.T_layer(x_flat)  # [batch_size, input_dim]
        
        # 第二层：B线性层（频域可训练变换）
        x_freq_transformed = self.B_layer(x_freq)  # [batch_size, input_dim]
        
        # 第一层：F线性层（频域到空域变换）
        output = self.F_layer(x_freq_transformed)  # [batch_size, input_dim]
        
        # 恢复原始形状
        output = output.view(*batch_dims, self.input_dim)
        
        return output
        

    
    def freeze_transform_matrices(self):
        """
        确认F和T线性层的冻结状态
        """
        # 确保F和T层的权重不需要梯度
        self.F_layer.weight.requires_grad = False
        self.T_layer.weight.requires_grad = False
        
        print(f"F层权重已冻结，形状: {self.F_layer.weight.shape}, requires_grad: {self.F_layer.weight.requires_grad}")
        print(f"T层权重已冻结，形状: {self.T_layer.weight.shape}, requires_grad: {self.T_layer.weight.requires_grad}")
        print(f"B层权重可训练，形状: {self.B_layer.weight.shape}, requires_grad: {self.B_layer.weight.requires_grad}")
        print(f"可训练参数数量: {sum(p.numel() for p in self.B_layer.parameters())}")
    
    def get_trainable_parameters(self):
        """
        获取可训练参数（只有B层）
        
        Returns:
            list: 可训练参数列表
        """
        return list(self.B_layer.parameters())
    
    def get_frequency_weights(self):
        """
        获取频域权重矩阵B
        
        Returns:
            torch.Tensor: B矩阵权重
        """
        return self.B_layer.weight.data
    
    def get_transform_matrices(self):
        """
        获取变换矩阵F和T（用于分析）
        
        Returns:
            tuple: (F_matrix, T_matrix)
        """
        return self.F_layer.weight.data.t(), self.T_layer.weight.data.t()





if __name__ == "__main__":
    # 测试代码
    print("测试傅立叶扩展模块...")
    
    # 创建模块实例（方阵模式）
    module = FourierExpansionModule(input_dim=64)
    
    # 创建测试输入
    x = torch.randn(10, 64)  # batch_size=10, input_dim=64
    
    # 前向传播
    output = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试参数冻结
    module.freeze_transform_matrices()
    trainable_params = module.get_trainable_parameters()
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    # 验证矩阵正交性
    print("\n验证矩阵正交性...")
    T_matrix = module.T_layer.weight.data  # T = C^T
    F_matrix = module.F_layer.weight.data  # F = C
    
    # 验证 T^T * T = I（T是正交的）
    orthogonality_check = torch.mm(T_matrix.t(), T_matrix)
    identity_error = torch.norm(orthogonality_check - torch.eye(module.input_dim))
    print(f"T矩阵正交性误差: {identity_error.item():.6f}")
    
    # 验证F=T^T关系
    ft_relation_error = torch.norm(F_matrix - T_matrix.t())
    print(f"F=T^T关系误差: {ft_relation_error.item():.6f}")
    
    # 测试掩码功能
    print("\n测试掩码功能...")
    mask = torch.zeros(64, 64)
    mask[:32, :32] = 1.0  # 只允许前32x32的参数更新
    
    masked_module = FourierExpansionModule(input_dim=64, mask=mask)
    
    # 保存初始权重
    initial_weight = masked_module.B_layer.weight.data.clone()
    
    # 执行一次前向传播和反向传播
    output = masked_module(x)
    loss = output.sum()
    loss.backward()
    
    # 手动更新参数（模拟优化器步骤）
    with torch.no_grad():
        masked_module.B_layer.weight.data -= 0.01 * masked_module.B_layer.weight.grad
    
    # 检查被掩码位置的参数是否保持不变
    weight_diff = masked_module.B_layer.weight.data - initial_weight
    masked_positions_changed = torch.norm(weight_diff[32:, :]).item()
    unmasked_positions_changed = torch.norm(weight_diff[:32, :32]).item()
    
    print(f"被掩码位置参数变化: {masked_positions_changed:.6f}")
    print(f"未被掩码位置参数变化: {unmasked_positions_changed:.6f}")
    print(f"掩码功能正常工作: {masked_positions_changed < 1e-6}")
    
    # 测试MaskMatrix类
    print("\n测试MaskMatrix类...")
    
    # 创建一个8x8的掩码矩阵管理器，每次分配10个位置
    mask_manager = MaskMatrix(matrix_shape=(8, 8), allocation_size=10)
    
    print(f"初始状态: {mask_manager.get_allocation_info()}")
    
    # 创建第一个掩码
    mask1 = mask_manager.create_mask()
    print(f"第一次分配后: {mask_manager.get_allocation_info()}")
    print(f"第一个掩码:\n{mask1}")
    
    # 创建第二个掩码
    mask2 = mask_manager.create_mask()
    print(f"第二次分配后: {mask_manager.get_allocation_info()}")
    print(f"第二个掩码:\n{mask2}")
    
    # 验证掩码不重叠
    overlap = torch.sum(mask1 * mask2)
    print(f"掩码重叠检查（应为0）: {overlap.item()}")
    
    # 显示当前状态矩阵
    print(f"当前状态矩阵:\n{mask_manager.get_state_matrix()}")
    
    # 尝试分配超出可用位置的掩码
    try:
        for i in range(10):  # 尝试多次分配直到空间不足
            new_mask = mask_manager.create_mask()
            print(f"第{i+3}次分配成功")
    except RuntimeError as e:
        print(f"预期的错误: {e}")
    
    print(f"最终状态: {mask_manager.get_allocation_info()}")