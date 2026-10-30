# AdamW 优化算法实验

《优化与博弈》课程论文的配套实验代码和测量数据。

本仓库包括二维二次函数仿真，以及在 Tiny Shakespeare 上训练小型自回归 Transformer 的实验。训练模型为 824,320 参数，实验用于研究优化机制与不同衰减设置的行为。

## 实验设置

- 模型：4 层 Pre-LayerNorm Transformer，宽度 128、4 个注意力头、FFN 512，上下文 128，batch size 32，残差 dropout 0.1。
- 数据：Tiny Shakespeare，连续切分为训练 90%、验证 5%、测试 5%；词表只由训练段建立，大小 65。
- 六组：无衰减 Adam；AdamW 衰减 0.01、0.1；耦合 L2 Adam 衰减 1e-5、0.001、0.1。
- 所有组使用 betas=(0.9,0.95)、eps=1e-8，统一参数分组；只衰减线性层权重，包含输出头。
- 预实验：种子 0，每组比较峰值学习率 3e-4 和 1e-3，各训练 200 步，以验证损失选学习率，共 12 次。
- 主实验：种子 11、22、33，每组每种子 1000 步，共 18 次；每次 4,096,000 个训练字符目标。
- 数据梯度先裁剪至全局范数 1；耦合 L2 由 Adam 内部在裁剪后加入。学习率为 5% 预热和余弦下降。
- 验证/测试分别评估全部完整非重叠块，共 55,680 个字符目标；测试只评估最终模型，不用于选学习率或检查点。

## 已测结果

以下为三个种子的测试交叉熵（nats/char），均值 ± 样本标准差，越低越好。困惑度及逐次数据见 `results/runs.csv`。

| 配置 | 测试 CE |
|---|---:|
| Adam，无衰减 | 1.9706 ± 0.0042 |
| AdamW，0.01 | 1.9712 ± 0.0041 |
| AdamW，0.1 | 1.9751 ± 0.0048 |
| Adam+L2，1e-5 | 1.9735 ± 0.0034 |
| Adam+L2，0.001 | 2.1030 ± 0.0147 |
| Adam+L2，0.1 | 2.5491 ± 0.0016 |

当前短程实验未显示 AdamW 优势；强耦合 L2 抑制了学习，弱 L2 接近基线。结果只针对当前模型、数据和预算，不能外推为超大模型训练结论，也不构成所有方法充分调参后的性能排名。

二维仿真冻结预条件矩阵、忽略动量，验证解耦与耦合衰减的不同驻点及简化模型的收敛上界。它不是对完整自适应 AdamW 收敛性的数值证明。

## 文件说明

```text
code/
  train.py                  模型、数据切分、学习率预实验与主训练
  analyze.py                二维仿真、统计汇总和实验绘图
  checks.py                 更新公式、零衰减等价、因果遮罩和参数分组检查
  recheck_saved_models.py   可选：从重新生成的模型权重复核测试损失
data/
  input.txt                 原始教学语料，运行无需在线下载
  README.md                 来源、SHA-256、切分边界
results/
  main/                     18 次正式运行的 summary.json 和 metrics.jsonl
  pilot/                    12 次预实验的 summary.json 和 metrics.jsonl
  figures/                  三幅实验图（PNG）
  runs.csv                  每次正式运行结果
  pilot_comparison.csv      学习率候选结果
  aggregate.json            均值、样本标准差、种子内配对差值
  dataset.json              语料哈希、词表和切分记录
  selected_lr.json           验证集选择的学习率
  quadratic.csv/.json        二维轨迹、上界和解析解
  checkpoint_verification.json  原始本机模型的独立复核记录
requirements.txt
run_all.ps1
.gitignore
README.md
```

`metrics.jsonl` 保存逐次评估记录；`summary.json` 保存最终结果、训练配置和实际运行环境。原始测量数据保留不变。

## 运行环境与复现

实测环境：Windows、Python 3.9.25、PyTorch 2.8.0+cu126、NumPy 1.26.3、Matplotlib 3.9.4，RTX 4080 Laptop GPU（12 GB）。GPU 使用 BF16 autocast，参数与优化器矩状态为 FP32。CPU 也可运行，但更慢；不同设备可能产生数值差异。

先准备 Python 环境，并根据设备安装兼容的 PyTorch；其余实测版本列于 `requirements.txt`。已有环境无需重复安装。

在仓库根目录执行以下命令即可检查实现并从已有日志重建全部图表，不重新训练：

```powershell
python code/checks.py
python code/analyze.py
```

完整流程：

```powershell
python code/train.py --stage pilot
python code/train.py --stage main
python code/analyze.py
```

或 PowerShell 一键执行 `./run_all.ps1`；指定解释器示例：

```powershell
./run_all.ps1 -PythonExe 'C:\Users\ROG\.conda\envs\ai\python.exe'
```

**默认会复用已完成结果。** 如要真正从零重新训练，先将现有 `results/` 改名备份，再执行完整流程。脚本在存在 `summary.json` 时跳过对应运行。修改模型结构或训练协议时也应备份旧结果，以免误用缓存。CPU 运行需分别向训练命令添加 `--device cpu`，或向一键脚本添加 `-Device cpu`。

训练将生成 `final_model.pt`；仓库不包含模型权重，`.gitignore` 也已排除它们。重新训练后可运行 `python code/recheck_saved_models.py` 核对所有最终模型。该脚本只应加载本项目自行生成、可信的检查点。仓库里的 `checkpoint_verification.json` 是原始本机 18 个模型的复核记录。

## 数据与参考来源

- [Tiny Shakespeare 数据来源](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)：数据文件及哈希见 `data/README.md`。
- [Adam](https://arxiv.org/abs/1412.6980)，Kingma 与 Ba，ICLR 2015。
- [AdamW](https://arxiv.org/abs/1711.05101)，Loshchilov 与 Hutter，ICLR 2019。
- [PyTorch 2.8 AdamW 文档](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html)。
- [PyTorch 2.8 Adam 文档](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html)：耦合 L2 显式使用 `decoupled_weight_decay=False`。

模型与实验脚本由本项目实现；教学语料来自以上公开来源。