# Qwen2.5 模型知识蒸馏实验：对比不同KL散度策略

## 简介

本项目旨在探索和比较不同的白盒知识蒸馏策略，特别是不同类型的KL散度损失函数，在大型语言模型上的效果。我们使用 `unsloth/Qwen2.5-7B-Instruct` 作为教师模型，将其知识蒸馏到较小的 `unsloth/Qwen2.5-3B-Instruct` 学生模型上。实验在 `yahma/alpaca-cleaned` 数据集的前1000条样本上进行，并使用 Google 的 IFEval (Instruction Following Eval) 进行评估。

## 模型

### 教师模型
-   **名称**: `unsloth/Qwen2.5-7B-Instruct`
-   **训练超参数**:
    -   Effective batch size: 8
    -   Num_train_epochs: 1
    -   Warmup_ratio: 0.05
    -   Learning_rate: 1e-4
    -   Weight_decay: 0.01

### 学生模型
-   **名称**: `unsloth/Qwen2.5-3B-Instruct`

## 数据集

-   **名称**: `yahma/alpaca-cleaned`
-   **使用子集**: 前 1000 条样本

## 评估

-   **评估工具**: Google IFEval (Instruction Following Evaluation)
-   **主要指标**:
    -   `prompt_level_strict_accuracy`
    -   `inst_level_strict_accuracy`
    -   `prompt_level_loose_accuracy`
    -   `inst_level_loose_accuracy`

## 知识蒸馏设置

### 通用蒸馏超参数 (应用于所有学生模型策略)
-   `distill_use_ce_loss`: True (即同时使用标准的交叉熵损失和KL散度损失)
-   `distill_kl_loss_weight`: 0.5 (KL散度损失在总损失中的权重)
-   `distill_kl_temperature`: 2.0 (知识蒸馏中的温度参数)
-   `distill_epochs`: 3
-   `distill_batch_size`: 1 (实际运行时调整以适应显存)
-   `distill_grad_accum`: 32 (Effective batch size for distillation: 32)
-   `distill_lr`: 5e-4 (学生模型蒸馏时的学习率)

### KL散度策略特定参数
-   **前向KL (fkl)**: 无特定额外参数
-   **反向KL (rkl)**: 无特定额外参数
-   **偏向前KL (skewed-fkl)**: `skew_lambda_fkl = 0.1`
-   **偏向反KL (skewed-rkl)**: `skew_lambda_rkl = 0.1`

## 实验结果

下表展示了教师模型以及采用不同KL散度蒸馏策略的学生模型的IFEval评估结果：

| 模型/策略         | prompt_level_strict_accuracy | inst_level_strict_accuracy | prompt_level_loose_accuracy | inst_level_loose_accuracy |
|-------------------|------------------------------|----------------------------|-----------------------------|---------------------------|
| **教师模型 (7B)** | 0.5800                       | 0.6687                     | 0.6100                      | 0.6994                    |
| 学生 (3B) - fkl   | 0.4769                       | 0.5695                     | 0.4991                      | 0.5923                    |
| 学生 (3B) - rkl   | **0.5915**                   | **0.6715**                 | **0.6285**                  | **0.7002**                |
| 学生 (3B) - skewed-fkl | 0.3697                       | 0.4796                     | 0.4048                      | 0.5132                    |
| 学生 (3B) - skewed-rkl | 0.4085                       | 0.5096                     | 0.4473                      | 0.5420                    |

### 结果分析

-   **教师模型性能**: 教师模型 `unsloth/Qwen2.5-7B-Instruct` 在IFEval上表现良好，为学生模型的蒸馏效果提供了基准。
-   **反向KL (rkl) 表现突出**: 采用反向KL散度进行蒸馏的学生模型 (`unsloth/Qwen2.5-3B-Instruct`) 在所有IFEval指标上均取得了最佳性能。值得注意的是，在 `inst_level_strict_accuracy` (0.6715 vs 0.6687)、`prompt_level_loose_accuracy` (0.6285 vs 0.6100) 和 `inst_level_loose_accuracy` (0.7002 vs 0.6994) 指标上，学生模型甚至略微超过了教师模型。这表明反向KL散度在该实验设置下是一种非常有效的知识迁移方法。
-   **前向KL (fkl) 表现**: 前向KL散度策略下的学生模型性能优于两种偏向KL策略，但逊于反向KL策略和教师模型。
-   **偏向KL (skewed-fkl & skewed-rkl) 表现**:
    -   `skewed-rkl` 策略 (使用 `skew_lambda_rkl = 0.1`) 的表现优于 `skewed-fkl` 策略。
    -   尽管如此，两种偏向KL策略在本实验中的性能均显著低于标准的 `fkl` 和 `rkl` 策略，也远低于教师模型。这提示 `skew_lambda` 参数为0.1时，这两种偏向策略可能不是最优选择，或者策略本身可能需要进一步的调整和实验。

## 结论

在本实验中，针对将 `unsloth/Qwen2.5-7B-Instruct` 的知识蒸馏到 `unsloth/Qwen2.5-3B-Instruct` 的任务，并使用 `yahma/alpaca-cleaned` 数据集的前1000条样本进行训练：
-   **反向KL散度 (rkl)** 是最有效的蒸馏策略，其学生模型在IFEval上的表现甚至在多个指令遵循能力指标上超越了教师模型。
-   标准的**前向KL散度 (fkl)** 表现尚可，优于偏向KL散度策略，但与反向KL相比有明显差距。
-   **偏向KL散度 (skewed-fkl, skewed-rkl)** 在当前参数设置 (`skew_lambda = 0.1`) 下效果不理想，其中 `skewed-rkl` 略优于 `skewed-fkl`，但两者均有待进一步研究和参数调优。

这表明在白盒知识蒸馏中，KL散度损失函数的选择及其具体参数配置对最终学生模型的性能有显著影响。