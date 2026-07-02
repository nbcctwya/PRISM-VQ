# Dataset 构建说明

本文档说明如何从本地 Qlib `.bin` 数据和 JKP 因子 CSV 生成 PRISM-VQ 训练所需的 `dataset/data/` 文件。

## 前提条件

1. 已安装项目依赖，尤其是 `pyqlib`、`pandas`、`numpy`、`torch`。
2. 已准备好 Qlib 日频数据，默认路径如下：

   ```text
   ~/.qlib/qlib_data/cn_data
   ~/.qlib/qlib_data/us_data
   ```

   对应配置文件位于：

   ```text
   dataset/2024_csi300.yaml
   dataset/2024_csi500.yaml
   dataset/2024_sp500.yaml
   ```

   其中 `qlib_init.provider_uri` 控制 Qlib 数据路径。

3. 已准备好 JKP 因子 CSV，并放在 `dataset/data/` 下：

   ```text
   dataset/data/[chn]_[all_themes]_[daily]_[vw_cap].csv
   dataset/data/[usa]_[all_themes]_[daily]_[vw_cap].csv
   ```

   中国市场的 `csi300`、`csi500` 使用 `chn` 文件；美国市场的 `sp500` 使用 `usa` 文件。

## 生成命令

在项目根目录执行：

```bash
python dataset/get_dataset.py --universe csi300
python dataset/get_dataset.py --universe csi500
python dataset/get_dataset.py --universe sp500
```

脚本会自动读取对应的 YAML 配置：

```text
csi300 -> dataset/2024_csi300.yaml
csi500 -> dataset/2024_csi500.yaml
sp500  -> dataset/2024_sp500.yaml
```

## 输出文件

生成结果会保存到：

```text
dataset/data/CN/
dataset/data/US/
```

以 `csi300`、`step_len=20`、预测 horizon 数量为 10 为例，输出文件包括：

```text
dataset/data/CN/csi300_20_dataframe_learn.pkl
dataset/data/CN/csi300_20_dataframe_infer.pkl
dataset/data/CN/csi300_20_h10_dl2_train.pkl
dataset/data/CN/csi300_20_h10_dl2_valid.pkl
dataset/data/CN/csi300_20_h10_dl2_test.pkl
dataset/data/CN/csi300_20_h10_dl2_dataset.pkl
```

这些文件会被 `stage1.py` 和 `stage2.py` 读取。主配置中的路径为：

```yaml
data:
  data_path: dataset/data
```

## 数据处理逻辑

`dataset/get_dataset.py` 会执行以下步骤：

1. 初始化 Qlib 数据源。
2. 从 JKP CSV 中筛选对应市场的日频市值加权因子。
3. 将 JKP 因子收益转换为 20 日滚动累计收益，并按 Qlib 交易日历对齐。
4. 使用 `Alpha158WithJKP` 生成股票特征、先验因子和未来收益标签。
5. 构建 `TSDatasetH`，并切分为：

   ```text
   train: 2009-01-01 到 2019-12-31
   valid: 2020-01-01 到 2021-12-31
   test:  2022-01-01 到 2024-12-31
   ```

6. 将训练、验证、测试数据序列化为 pickle 文件。

## 注意事项

- `dataset/data/` 已被 `.gitignore` 排除，不会提交到 Git。
- 如果缺少 JKP CSV，脚本会报 `JKP factor file not found`。
- 如果 Qlib 路径不同，修改对应 YAML 中的 `qlib_init.provider_uri`。
- 如果重新生成数据集，旧的同名 pickle 会被覆盖。
- 建议在服务器镜像中固定 Qlib `.bin` 数据，在每台服务器本地重新生成 `dataset/data/`。
