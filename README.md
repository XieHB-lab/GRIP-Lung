# GAN-based Gene Expression Prediction

This repository provides a PyTorch implementation of a **conditional Generative Adversarial Network (GAN)** for gene expression prediction under different drug and cell line conditions.

The framework is designed to model the mapping between baseline gene expression profiles and perturbed expression states, and can be easily adapted to different transcriptomic datasets.

---

## Overview

Gene expression prediction under varying biological conditions is a fundamental task in computational biology and drug response modeling.
In this work, we implement a **GAN-based regression framework**, where:

* The **Generator** predicts target gene expression profiles conditioned on:

  * baseline gene expression
  * drug identity
  * cell line identity
* The **Discriminator** distinguishes real from generated expression profiles under the same conditions

The model is trained using a combination of **reconstruction loss (MSE)** and **adversarial loss**, and evaluated via **K-fold cross-validation**.

---

## Repository Structure

```text
.
├── data/
│   └── README.txt          # Description of expected data format
├── data_utils.py           # Data loading and preprocessing utilities
├── models.py               # Generator and discriminator model definitions
├── metrics.py              # Evaluation metrics
├── train.py                # GAN training and cross-validation logic
├── run_experiment.py       # Main script for running experiments
├── .gitignore
└── README.md
```

---

## Data Format

The model expects gene expression matrices in CSV format.

* `input_expression.csv`: baseline gene expression profiles
* `output_expression.csv`: target gene expression profiles

Both files follow the same structure:

* Rows represent genes
* Columns represent samples
* Additional rows encode sample, cell line, and drug information

Detailed format specifications can be found in `data/README.txt`.

> ⚠️ Large raw datasets are **not included** in this repository.
> Users should place their own data files under the `data/` directory.

---

## Requirements

* Python >= 3.8
* PyTorch
* NumPy
* Pandas
* scikit-learn

Install dependencies via:

```bash
pip install torch numpy pandas scikit-learn
```

---

## Tested environment
The code was tested with the CPU version of PyTorch on Windows.

---

## Usage

1. Prepare your data files and place them in the `data/` directory.
2. Update the data paths if necessary in `run_experiment.py`.
3. Run the experiment:

```bash
python run_experiment.py
```

The script performs model training, cross-validation, and evaluation automatically.

---

## Output

The training script outputs:

* Prediction performance metrics (e.g., RMSE, R², classification-based scores)
* Discriminator accuracy and AUC
* A CSV file summarizing the results for the tested model

---

## Notes

* This repository focuses on **method implementation and reproducibility**.
* Hyperparameters are set to reasonable defaults and can be adjusted in `train.py`.
* The code is modular and can be extended with alternative generators, discriminators, or evaluation metrics.

---

## Citation

If you find this code useful, please consider citing our work.

---

## Contact

For questions or suggestions, please open an issue or contact the repository owner.

