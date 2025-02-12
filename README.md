# WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting

This is the official implementation of the **pure AR/WAVE Transformers**, introduced in the paper [WAVE Attention Mechanisms](https://arxiv.org/abs/2410.03159). The core model is located in `models/Autoregressive.py`. The repository includes various attention mechanisms, with aliases corresponding to each method used in the provided code and scripts as follows:

```
# FullAttention (CausalSelfAttention): Standard Softmax Attention
# FullAttentionARMA (CausalSelfAttentionARMA): Standard Softmax Attention + ARMA
# TwoStageSelfgatingRNN: Element-wise Linear Attention (AFT)
# TwoStageSelfgatingRNNARMA: Element-wise Linear Attention + ARMA
# LinearAttention: Linear Attention
# LinearAttentionARMA: Linear Attention + ARMA
# MaskedLinear: Fixed Attention
# MaskedLinearARMA: Fixed Attention + ARMA
# GatedLinearAttention: Gated Linear Attention
# GatedLinearAttentionARMA: Gated Linear Attention + ARMA
# MovingAverageGatedAttention: MEGA
```

## 1. Install Required Packages

First, install PyTorch with GPU support by following the instructions on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally). Then, install the additional dependencies with:

```bash
pip install -r requirements.txt
```

## 2. Download Datasets

You can download the 12 datasets used in the paper from the link provided by [itransformer](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view) [1]. Place the downloaded files in `dataset/`.

## 3. Run the Training Scripts

To train the AR/ARMA Transformers, run:

```bash
bash autoregressive.sh
```

For baseline models used in the paper, run:

```bash
bash baseline.sh
```

## 4. Track the Training Process

You can visualize the training process using TensorBoard by running the following command:

```bash
nohup tensorboard --logdir runs --port 6006 --bind_all > tensorb.log 2>&1 &
```

## 5. Training on Custom Data

To train the models on your own dataset, format the CSV file with the first column as `date` (timestamps), and the remaining columns as the time series values. Place your dataset in the `dataset/` folder. 

Next, update the following arrays in the scripts to include your dataset:

```bash
data_names=("weather/weather.csv" "ETT-small/ETTm1.csv" "ETT-small/ETTm2.csv" "ETT-small/ETTh1.csv" "ETT-small/ETTh2.csv" "Solar/solar_AL.txt" "electricity/electricity.csv" "PEMS/PEMS03.npz" "PEMS/PEMS04.npz" "PEMS/PEMS07.npz" "PEMS/PEMS08.npz" "traffic/traffic.csv")
data_alias=("Weather" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "Solar" "ECL" "PEMS03" "PEMS04" "PEMS07" "PEMS08" "Traffic")
data_types=("custom" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "Solar" "custom" "PEMS" "PEMS" "PEMS" "PEMS" "custom")
enc_ins=(21 7 7 7 7 137 321 358 307 883 170 862)  # Number of time series in each dataset
batch_sizes=(32 32 32 32 32 32 32 32 32 32 32 32)  # Batch sizes for each dataset
grad_accums=(1 1 1 1 1 1 1 1 1 1 1 1)  # Gradient accumulation steps
```

Modify these lists to match the configuration of your custom dataset.

## References

[1] Liu, Yong, et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." arXiv preprint arXiv:2310.06625 (2023).
