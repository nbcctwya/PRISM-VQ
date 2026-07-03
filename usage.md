# PRISM-VQ 使用说明

本文档记录本仓库从数据集构建、Stage 1 训练、Stage 2 训练到 Qlib 回测的常用流程。下面的命令默认在项目根目录执行，并使用 conda 环境 `prism-vq`。

```bash
cd /home/nbcctwya/quant/PRISM-VQ
```

也可以先激活环境：

```bash
conda activate prism-vq
```

如果不想激活环境，所有命令都可以写成：

```bash
conda run -n prism-vq python ...
```

## 1. 构建数据集

数据集由 Qlib 日频 `.bin` 数据和 JKP 因子 CSV 共同生成。默认需要以下 Qlib 数据路径：

```text
~/.qlib/qlib_data/cn_data
~/.qlib/qlib_data/us_data
```

还需要把 JKP 因子文件放在：

```text
dataset/data/[chn]_[all_themes]_[daily]_[vw_cap].csv
dataset/data/[usa]_[all_themes]_[daily]_[vw_cap].csv
```

生成 CSI300 数据集：

```bash
conda run -n prism-vq python dataset/get_dataset.py --universe csi300
```

生成 S&P 500 数据集：

```bash
conda run -n prism-vq python dataset/get_dataset.py --universe sp500
```

可选的 CSI500：

```bash
conda run -n prism-vq python dataset/get_dataset.py --universe csi500
```

脚本会读取对应配置：

```text
csi300 -> dataset/2025_csi300.yaml
sp500  -> dataset/2025_sp500.yaml
csi500 -> dataset/2025_csi500.yaml
```

输出位置：

```text
dataset/data/CN/
dataset/data/US/
```

以 CSI300 为例，主要输出包括：

```text
dataset/data/CN/csi300_20_dataframe_learn.pkl
dataset/data/CN/csi300_20_dataframe_infer.pkl
dataset/data/CN/csi300_20_h10_dl2_train.pkl
dataset/data/CN/csi300_20_h10_dl2_valid.pkl
dataset/data/CN/csi300_20_h10_dl2_test.pkl
dataset/data/CN/csi300_20_h10_dl2_dataset.pkl
```

注意：

- `dataset/data/` 不提交到 Git。
- 如果 Qlib 路径不同，修改 `dataset/2025_*.yaml` 里的 `qlib_init.provider_uri`。
- 如果重新生成数据集，同名 pickle 会被覆盖。

## 2. 运行 Stage 1

Stage 1 训练 VQ-VAE 表征模型。脚本入口是 `stage1.py`，配置来自 `configs/config.yaml`。Stage 1 固定使用 seed 42。

运行 CSI300：

```bash
WANDB_MODE=offline MPLCONFIGDIR=/tmp/matplotlib \
conda run -n prism-vq python stage1.py data.universe=csi300
```

运行 S&P 500：

```bash
WANDB_MODE=offline MPLCONFIGDIR=/tmp/matplotlib \
conda run -n prism-vq python stage1.py data.universe=sp500
```

常用输出位置：

```text
checkpoints/
checkpoints/wandb/
```

Stage 1 最重要的输出是 `checkpoints/` 下的 VQ-VAE checkpoint，例如：

```text
checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=9-val_loss=0.5682.ckpt
checkpoints/infusp500_h128_VQK512_C128_emb128_dl2p10_s42-epoch=1-val_loss=0.7590.ckpt
```

后续 Stage 2 需要把这个文件名填到 `predictor.saved_model`。

## 3. 运行 Stage 2

Stage 2 使用 Stage 1 checkpoint，训练收益预测模型，并在测试集上输出预测分数和 IC 指标。脚本入口是 `stage2.py`。

### CSI300 单 seed 示例

```bash
WANDB_MODE=offline MPLCONFIGDIR=/tmp/matplotlib \
conda run -n prism-vq python stage2.py \
  data.universe=csi300 \
  predictor.saved_model=infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=9-val_loss=0.5682.ckpt \
  predictor.aux_weight=0.01 \
  predictor.n_expert=2 \
  predictor.transformer.num_heads=2 \
  train.seed=0
```

### S&P 500 单 seed 示例

```bash
WANDB_MODE=offline MPLCONFIGDIR=/tmp/matplotlib \
conda run -n prism-vq python stage2.py \
  data.universe=sp500 \
  predictor.saved_model=infusp500_h128_VQK512_C128_emb128_dl2p10_s42-epoch=1-val_loss=0.7590.ckpt \
  predictor.aux_weight=0.001 \
  predictor.n_expert=8 \
  predictor.transformer.num_heads=4 \
  train.seed=0
```

切换市场时，最容易漏改的是：

```text
data.universe
predictor.saved_model
predictor.aux_weight
predictor.n_expert
predictor.transformer.num_heads
```

其中 `predictor.k` 默认是 `${half:${predictor.n_expert}}`，通常不用手动改。

### 多 seed 运行

可以用 Hydra multirun 一次跑多个 seed：

```bash
WANDB_MODE=offline MPLCONFIGDIR=/tmp/matplotlib \
conda run -n prism-vq python stage2.py -m \
  data.universe=csi300 \
  predictor.saved_model=infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=9-val_loss=0.5682.ckpt \
  predictor.aux_weight=0.01 \
  predictor.n_expert=2 \
  predictor.transformer.num_heads=2 \
  train.seed=0,1,2,3,4
```

S&P 500 同理：

```bash
WANDB_MODE=offline MPLCONFIGDIR=/tmp/matplotlib \
conda run -n prism-vq python stage2.py -m \
  data.universe=sp500 \
  predictor.saved_model=infusp500_h128_VQK512_C128_emb128_dl2p10_s42-epoch=1-val_loss=0.7590.ckpt \
  predictor.aux_weight=0.001 \
  predictor.n_expert=8 \
  predictor.transformer.num_heads=4 \
  train.seed=0,1,2,3,4
```

Stage 2 输出包括两类：

```text
checkpoints/
res/
```

`checkpoints/` 保存 Stage 2 checkpoint；`res/` 保存测试集预测和指标。以 CSI300 为例：

```text
res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/
├── 0_best.pkl
├── 0_metric.csv
├── 1_best.pkl
├── 1_metric.csv
└── ...
```

其中：

- `*_best.pkl`：测试集每日每只股票的预测分数和 label。
- `*_metric.csv`：对应 seed 的 `IC`、`ICIR`、`RankIC`、`RankICIR`。

## 4. 生成 Ensemble 预测

如果已经有 5 个 seed 的 Stage 2 预测，可以把 `score` 按 `(datetime, instrument)` 对齐后取平均，生成 ensemble 预测文件。

CSI300：

```bash
conda run -n prism-vq python scripts/ensemble_predictions.py \
  --run-dir res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3 \
  --seeds 0 1 2 3 4
```

S&P 500：

```bash
conda run -n prism-vq python scripts/ensemble_predictions.py \
  --run-dir res/VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3 \
  --seeds 0 1 2 3 4
```

默认输出：

```text
res/.../ensemble_5seed_best.pkl
```

也可以显式指定输入和输出：

```bash
conda run -n prism-vq python scripts/ensemble_predictions.py \
  --pred res/run/0_best.pkl res/run/1_best.pkl res/run/2_best.pkl \
  --output res/run/ensemble_3seed_best.pkl
```

## 5. 运行 Qlib 回测

回测脚本是 `backtest_qlib.py`。它读取 Stage 2 的 `*_best.pkl` 或 ensemble pkl，用 Qlib 的 `TopkDropoutStrategy` 跑 TopK-DropN 回测。

默认论文口径：

```text
topk = 30
drop = 5
open_cost = 0.0005
close_cost = 0.0015
min_cost = 0
start_time = 2023-01-01
end_time = 2025-12-31
```

CSI300 seed0：

```bash
conda run -n prism-vq python backtest_qlib.py \
  --universe csi300 \
  --pred_path res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/0_best.pkl \
  --topk 30 \
  --drop 5 \
  --output_dir res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/backtest/seed0_top30_drop5
```

CSI300 ensemble：

```bash
conda run -n prism-vq python backtest_qlib.py \
  --universe csi300 \
  --pred_path res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/ensemble_5seed_best.pkl \
  --topk 30 \
  --drop 5 \
  --output_dir res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/backtest/ensemble_5seed_top30_drop5
```

S&P 500 ensemble：

```bash
conda run -n prism-vq python backtest_qlib.py \
  --universe sp500 \
  --pred_path res/VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/ensemble_5seed_best.pkl \
  --topk 30 \
  --drop 5 \
  --output_dir res/VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/backtest/ensemble_5seed_top30_drop5
```

如果想测试其他 K，例如 `K=40`、`K=50`，只需要改 `--topk` 和输出目录名：

```bash
conda run -n prism-vq python backtest_qlib.py \
  --universe csi300 \
  --pred_path res/.../ensemble_5seed_best.pkl \
  --topk 40 \
  --drop 5 \
  --output_dir res/.../backtest/ensemble_5seed_top40_drop5
```

零交易成本检查：

```bash
conda run -n prism-vq python backtest_qlib.py \
  --universe csi300 \
  --pred_path res/.../ensemble_5seed_best.pkl \
  --topk 30 \
  --drop 5 \
  --open_cost 0 \
  --close_cost 0 \
  --min_cost 0 \
  --output_dir res/.../backtest/ensemble_5seed_top30_drop5_zero_cost_check
```

回测输出目录包含：

```text
portfolio_metric.csv
portfolio_return.csv
qlib_report.csv
qlib_signal.pkl
positions.pkl
```

各文件含义：

- `portfolio_metric.csv`：回测指标。重点看 `project_portfolio` 和 `project_excess` 两列。
- `portfolio_return.csv`：逐日收益。`return` 是扣成本后的净收益，`gross_return` 是 Qlib 原始收益，`cost` 是交易成本。
- `qlib_report.csv`：Qlib 原生 report。
- `qlib_signal.pkl`：标准化后的 Qlib signal。
- `positions.pkl`：每日持仓。

## 6. 结果汇总

Stage 2 的预测指标在：

```text
res/.../0_metric.csv
res/.../1_metric.csv
...
```

回测指标在：

```text
res/.../backtest/<run_name>/portfolio_metric.csv
```

当前已经生成过的主汇总表包括：

```text
res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/csi300_experiment_summary_top30_drop5.csv
res/VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3/sp500_experiment_summary_top30_drop5.csv
```

其中 ensemble 行的 `ic/icir/ric/ricir` 使用 5 个 seed 指标的算术平均；回测列使用 ensemble 预测文件单独回测得到的结果。

## 7. 常见注意事项

- Stage 1 和 Stage 2 都依赖 `configs/config.yaml`，推荐用 Hydra override 记录每次实验配置。
- Stage 2 的 `predictor.saved_model` 可以写 checkpoint 文件名；脚本会默认到 `checkpoints/` 下寻找。
- 从 CSI300 切到 S&P 500 时，务必同步修改 `data.universe`，否则会使用错误的股票池和数据路径。
- `portfolio_mdd` 是绝对组合净值最大回撤；`excess_mdd` 是相对 benchmark 的超额回撤。
- 回测脚本保存的 `return` 是扣除交易成本后的净收益；若要核对成本影响，可以同时查看 `gross_return` 和 `cost`。
- 建议重要回测都显式指定 `--output_dir`，避免覆盖或混淆已有结果。
