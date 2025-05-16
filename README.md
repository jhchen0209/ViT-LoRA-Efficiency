# ViT-LoRA 效率比較實驗

這個專案使用 Oxford-IIIT Pet 資料集進行實驗，該資料集是一個針對貓與狗的圖像分類任務， 37 種不同品種，共約 7,349 張圖像。在本專案中，採用 Vision Transformer (ViT) 架構，並載入 DINO ViT-S/16 的預訓練權重（facebook/dino-vits16） 作為 backbone 進行微調。實驗中手動實作了 LoRA（Low-Rank Adaptation） 機制，並比較了以下三種不同的微調策略：
1. 完整微調 (Full Fine-tuning)
2. LoRA 微調 (LoRA Fine-tuning)
3. 僅分類頭訓練 (Classification Head Only)


## 實驗方法

### 1. 完整微調 (Full Fine-tuning)
- 訓練所有模型參數
- 學習率：1e-4
- 批次大小：128
- 訓練輪數：20 epochs

### 2. LoRA 微調 (LoRA Fine-tuning)
- 使用 LoRA 技術微調注意力層
- 只訓練 LoRA 參數和分類頭
- 學習率：5e-5
- 批次大小：128
- 訓練輪數：20 epochs
- LoRA 設定：
  - r = 8
  - alpha = 16
  - dropout = 0.1

### 3. 僅分類頭訓練 (Classification Head Only)
- 凍結所有預訓練參數
- 只訓練分類頭
- 學習率：1e-4
- 批次大小：128
- 訓練輪數：20 epochs

## 模型架構
- 基礎模型：facebook/dino-vits16
- 輸入圖片大小：224x224
- 資料預處理：
  - Resize 到 224x224
  - 標準化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## 資料集
- 資料目錄：./DATA/images
- 訓練集/測試集比例：80/20
- 使用分層抽樣確保類別分布

## 訓練結果

### 完整微調
[Epoch 20] Train Loss: 0.0001, Acc: 1.0000 | Val Loss: 0.4457, Acc: 0.8978  
訓練完成，總耗時: 335.48 秒    

=== 模型參數統計 ===  
總參數數量: 21,679,909
可訓練參數: 21,679,909
可訓練比例: 100.00%

### LoRA 微調
[Epoch 20] Train Loss: 0.1255, Acc: 0.9751 | Val Loss: 0.2337, Acc: 0.9317  
訓練完成，總耗時: 304.65 秒  

=== 模型參數統計 ===  
總參數數量: 21,827,365
可訓練參數: 161,701
可訓練比例: 0.74%


### 僅分類頭訓練
[Epoch 20] Train Loss: 0.1485, Acc: 0.9650 | Val Loss: 0.2428, Acc: 0.9249  
訓練完成，總耗時: 155.48 秒  

=== 模型參數統計 ===  
總參數數量: 21,679,909
可訓練參數: 14,245
可訓練比例: 0.07%

## 環境需求
- Python 3.10+
- PyTorch
- transformers
- torchvision
- scikit-learn
- PIL
- pandas