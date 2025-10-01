#!/usr/bin/env python3
"""
测试傅立叶扩展模块

这个脚本测试FourierExpansionModule和AdaptiveFourierExpansion的功能
包括：
1. 基本功能测试
2. 参数冻结验证
3. 梯度流测试
4. 性能对比
5. 频率选择效果验证
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cdm.fourier_expansion import FourierExpansionModule, AdaptiveFourierExpansion


def test_basic_functionality():
    """
    测试基本功能
    """
    print("=== 基本功能测试 ===")
    
    # 创建模块
    module = FourierExpansionModule(input_dim=32, output_dim=64, num_frequencies=16)
    
    # 测试不同形状的输入
    test_cases = [
        (10, 32),      # 2D: [batch, features]
        (5, 8, 32),    # 3D: [batch, seq, features]
        (2, 4, 6, 32), # 4D: [batch, h, w, features]
    ]
    
    for i, shape in enumerate(test_cases):
        x = torch.randn(*shape)
        output = module(x)
        expected_shape = shape[:-1] + (64,)
        
        print(f"测试 {i+1}: 输入形状 {x.shape} -> 输出形状 {output.shape}")
        assert output.shape == expected_shape, f"形状不匹配: 期望 {expected_shape}, 得到 {output.shape}"
        print(f"✓ 形状正确")
    
    print("基本功能测试通过!\n")


def test_parameter_freezing():
    """
    测试参数冻结
    """
    print("=== 参数冻结测试 ===")
    
    module = FourierExpansionModule(input_dim=32, output_dim=64, num_frequencies=16)
    
    # 检查哪些参数需要梯度
    trainable_params = []
    frozen_params = []
    
    for name, param in module.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    print(f"可训练参数: {trainable_params}")
    print(f"冻结参数: {frozen_params}")
    
    # 检查buffer（应该包含F_matrix和T_matrix）
    buffers = [name for name, _ in module.named_buffers()]
    print(f"缓冲区（冻结矩阵）: {buffers}")
    
    # 验证只有B层是可训练的
    assert len(trainable_params) == 1, f"应该只有1个可训练参数，实际有 {len(trainable_params)}"
    assert 'B_layer.weight' in trainable_params, "B_layer.weight应该是可训练的"
    assert 'F_matrix' in buffers and 'T_matrix' in buffers, "F_matrix和T_matrix应该在buffers中"
    
    print("✓ 参数冻结正确")
    print("参数冻结测试通过!\n")


def test_gradient_flow():
    """
    测试梯度流
    """
    print("=== 梯度流测试 ===")
    
    module = FourierExpansionModule(input_dim=32, output_dim=64, num_frequencies=16)
    
    # 创建输入和目标
    x = torch.randn(10, 32, requires_grad=True)
    target = torch.randn(10, 64)
    
    # 前向传播
    output = module(x)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"输入梯度形状: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"B层权重梯度形状: {module.B_layer.weight.grad.shape if module.B_layer.weight.grad is not None else 'None'}")
    
    # 验证梯度存在
    assert x.grad is not None, "输入应该有梯度"
    assert module.B_layer.weight.grad is not None, "B层权重应该有梯度"
    
    # 验证F和T矩阵没有梯度（因为它们是buffer）
    assert not module.F_matrix.requires_grad, "F_matrix不应该需要梯度"
    assert not module.T_matrix.requires_grad, "T_matrix不应该需要梯度"
    
    print("✓ 梯度流正确")
    print("梯度流测试通过!\n")


def test_adaptive_module():
    """
    测试自适应模块
    """
    print("=== 自适应模块测试 ===")
    
    strategies = ['low', 'high', 'random', 'adaptive']
    x = torch.randn(5, 32)
    
    for strategy in strategies:
        module = AdaptiveFourierExpansion(
            input_dim=32, output_dim=64, num_frequencies=16,
            frequency_selection=strategy, selection_ratio=0.5
        )
        
        output = module(x)
        active_freq = module.frequency_mask.sum().item()
        
        print(f"策略 '{strategy}': 活跃频率 {active_freq}/{module.num_frequencies}")
        print(f"  输出形状: {output.shape}")
        
        # 验证活跃频率数量
        expected_active = int(module.num_frequencies * module.selection_ratio)
        assert active_freq == expected_active, f"活跃频率数量不匹配: 期望 {expected_active}, 得到 {active_freq}"
    
    print("✓ 自适应模块功能正确")
    print("自适应模块测试通过!\n")


def test_performance_comparison():
    """
    性能对比测试
    """
    print("=== 性能对比测试 ===")
    
    # 创建不同的模块进行对比
    input_dim, output_dim = 128, 256
    
    # 标准线性层
    linear = nn.Linear(input_dim, output_dim)
    
    # 傅立叶扩展模块（不同频率数量）
    fourier_32 = FourierExpansionModule(input_dim, output_dim, num_frequencies=32)
    fourier_64 = FourierExpansionModule(input_dim, output_dim, num_frequencies=64)
    fourier_128 = FourierExpansionModule(input_dim, output_dim, num_frequencies=128)
    
    # 自适应模块
    adaptive = AdaptiveFourierExpansion(
        input_dim, output_dim, num_frequencies=64,
        frequency_selection='adaptive', selection_ratio=0.5
    )
    
    modules = {
        'Linear': linear,
        'Fourier-32': fourier_32,
        'Fourier-64': fourier_64,
        'Fourier-128': fourier_128,
        'Adaptive': adaptive
    }
    
    # 测试输入
    x = torch.randn(100, input_dim)
    
    print("模块参数数量对比:")
    for name, module in modules.items():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {param_count:,} 参数")
    
    # 简单的时间测试
    import time
    
    print("\n前向传播时间对比 (100次平均):")
    for name, module in modules.items():
        module.eval()
        
        # 预热
        for _ in range(10):
            _ = module(x)
        
        # 计时
        start_time = time.time()
        for _ in range(100):
            _ = module(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        print(f"  {name}: {avg_time:.3f} ms")
    
    print("性能对比测试完成!\n")


def visualize_frequency_response():
    """
    可视化频率响应
    """
    print("=== 频率响应可视化 ===")
    
    try:
        # 创建模块
        module = FourierExpansionModule(input_dim=64, output_dim=64, num_frequencies=32)
        
        # 创建不同频率的输入信号
        t = torch.linspace(0, 2*np.pi, 64)
        frequencies = [1, 2, 4, 8]  # 不同频率的正弦波
        
        plt.figure(figsize=(15, 10))
        
        for i, freq in enumerate(frequencies):
            # 生成正弦波输入
            x = torch.sin(freq * t).unsqueeze(0)  # [1, 64]
            
            # 通过模块
            with torch.no_grad():
                output = module(x).squeeze(0)  # [64]
            
            # 绘制
            plt.subplot(2, 2, i+1)
            plt.plot(t.numpy(), x.squeeze().numpy(), 'b-', label=f'输入 (频率={freq})', linewidth=2)
            plt.plot(t.numpy(), output.numpy(), 'r--', label='输出', linewidth=2)
            plt.title(f'频率 {freq} 的响应')
            plt.xlabel('时间')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/jasangela/One-for-More/frequency_response.png', dpi=150, bbox_inches='tight')
        print("✓ 频率响应图已保存到 frequency_response.png")
        
    except ImportError:
        print("matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"可视化过程中出错: {e}")
    
    print("频率响应可视化完成!\n")


def test_mathematical_properties():
    """
    测试数学性质
    """
    print("=== 数学性质测试 ===")
    
    module = FourierExpansionModule(input_dim=32, output_dim=32, num_frequencies=16)
    
    # 测试线性性质
    x1 = torch.randn(5, 32)
    x2 = torch.randn(5, 32)
    a, b = 2.0, 3.0
    
    with torch.no_grad():
        # 线性组合的输出
        linear_combo_output = module(a * x1 + b * x2)
        
        # 输出的线性组合
        output_combo = a * module(x1) + b * module(x2)
        
        # 检查线性性质（应该近似相等）
        diff = torch.abs(linear_combo_output - output_combo).max().item()
        print(f"线性性质误差: {diff:.6f}")
        
        # 由于数值精度，允许小的误差
        assert diff < 1e-5, f"线性性质测试失败，误差过大: {diff}"
    
    # 测试零输入
    zero_input = torch.zeros(5, 32)
    zero_output = module(zero_input)
    zero_norm = torch.norm(zero_output).item()
    print(f"零输入的输出范数: {zero_norm:.6f}")
    
    # 由于B矩阵初始化为单位矩阵，零输入应该产生接近零的输出
    assert zero_norm < 1e-5, f"零输入测试失败，输出范数过大: {zero_norm}"
    
    print("✓ 数学性质正确")
    print("数学性质测试通过!\n")


def main():
    """
    运行所有测试
    """
    print("开始傅立叶扩展模块测试...\n")
    
    try:
        test_basic_functionality()
        test_parameter_freezing()
        test_gradient_flow()
        test_adaptive_module()
        test_performance_comparison()
        test_mathematical_properties()
        visualize_frequency_response()
        
        print("🎉 所有测试通过!")
        print("\n傅立叶扩展模块功能正常，可以使用。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()