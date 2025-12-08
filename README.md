# GRIP-Lung
## Installation

Clone this repository:

```bash
git clone https://github.com/yourname/GRIP-Lung.git
cd GRIP-Lung
Install dependencies:

bash

pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib
Make sure you have Python ≥ 3.8 and PyTorch ≥ 2.0 installed.

Usage
Place your dataset files (p-lung.csv and p-drug.csv) in a directory, e.g., E:\data\geo_data.

Update file paths in the script:

python

input_df  = pd.read_csv('E:\\data\\geo_data\\p-lung.csv', header=None)
output_df = pd.read_csv('E:\\data\\geo_data\\p-drug.csv', header=None)
Run the script:

bash

python cgan_gene_expression.py
Results:

CV Metrics: Printed per fold and averaged across folds.

Bar Plot: Saved as cv_gan_artifacts/metrics_barplot.png.
