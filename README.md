# GRIP-Lung
# cGAN Residual Generator for Gene Expression Prediction
This repository implements a **Conditional GAN (cGAN)** model for predicting post-treatment gene expression profiles from pre-treatment gene expression data. The model incorporates **FiLM-based residual generator** and a **lightweight MLP discriminator**, conditioning on cell line and drug information.
---
## Features
- **FiLM Residual Generator**: Enhances gene expression prediction using feature-wise linear modulation (FiLM) with residual connections.  
- **Lightweight MLP Discriminator**: Efficient adversarial training for distinguishing real vs. generated gene expression profiles.  
- **Conditional Input**: Supports embedding for both cell lines and drugs.  
- **Cross-Validation**: 5-fold CV evaluation with MSE, R², and binary classification metrics (Accuracy, Precision, Recall, F1).  
- **Visualization**: Bar plots for aggregated CV metrics with mean ± standard deviation.
---
## Dataset Format
The code expects two CSV files:  
1. **Input gene expression (`p-lung.csv`)**  
2. **Output gene expression (`p-drug.csv`)**  
## Installation
1. Clone this repository:
```bash
git clone https://github.com/XieHB-lab/GRIP-Lung.git
cd GRIP-Lung
2. Install dependencies:
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib
Make sure you have Python ≥ 3.8 and PyTorch ≥ 2.0 installed.
## Usage
Place your dataset files (p-lung.csv and p-drug.csv) in a directory, e.g., E:\data\geo_data.
Update file paths in the script:
input_df  = pd.read_csv('E:\\data\\geo_data\\p-lung.csv', header=None)
output_df = pd.read_csv('E:\\data\\geo_data\\p-drug.csv', header=None)
Run the script:
python cgan_gene_expression.py
Results:
CV Metrics: Printed per fold and averaged across folds.
Bar Plot: Saved as cv_gan_artifacts/metrics_barplot.png.
## Model Architecture
FiLM Residual Generator
Input: Pre-treatment gene expression (x_gene), cell ID, drug ID.
Cell and drug embeddings are concatenated and passed through FiLM layers.
Residual block refines the modulated features.
Output: Predicted post-treatment gene expression.
Lightweight MLP Discriminator
Input: Gene expression, cell ID, drug ID.
Predicts the probability of a gene expression profile being real or generated.
## Notes
Small datasets may lead to overfitting; consider augmenting data.
GPU is recommended for faster training.
