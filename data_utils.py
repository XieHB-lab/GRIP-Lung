import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(input_path, output_path, device):
    """
    Load and preprocess gene expression data for GAN training.

    Parameters
    ----------
    input_path : pathlib.Path or str
        Path to the input gene expression CSV file
        (e.g., pre-treatment or baseline expression).
    output_path : pathlib.Path or str
        Path to the output gene expression CSV file
        (e.g., post-treatment or target expression).
    device : torch.device
        Device on which tensors will be allocated (CPU or CUDA).

    Returns
    -------
    X : torch.Tensor
        Standardized input gene expression matrix of shape (N, G).
    y : torch.Tensor
        Standardized output gene expression matrix of shape (N, G).
    drug : torch.Tensor
        Encoded drug labels of shape (N, 1).
    cell : torch.Tensor
        Encoded cell line labels of shape (N, 1).
    gene_dim : int
        Number of genes (feature dimension).
    """

    # --------------------------------------------------------------
    # Load raw CSV files (no header assumed)
    # --------------------------------------------------------------
    input_df = pd.read_csv(input_path, header=None)
    output_df = pd.read_csv(output_path, header=None)

    # --------------------------------------------------------------
    # Parse metadata from the input file
    # Row 0: sample names
    # Row 1: cell line names
    # Row 2: drug names
    # Rows 3+: gene expression values
    # --------------------------------------------------------------
    sample_names = input_df.iloc[0, 1:].values
    cell_names   = input_df.iloc[1, 1:].values
    drug_names   = input_df.iloc[2, 1:].values
    gene_names   = input_df.iloc[3:, 0].values

    # --------------------------------------------------------------
    # Extract gene expression matrices
    # Transpose to shape: (num_samples, num_genes)
    # --------------------------------------------------------------
    gene_expr_input  = input_df.iloc[3:, 1:].astype(float).values.T
    gene_expr_output = output_df.iloc[3:, 1:].astype(float).values.T

    # --------------------------------------------------------------
    # Encode categorical variables (drug and cell line)
    # --------------------------------------------------------------
    le_drug = LabelEncoder()
    le_cell = LabelEncoder()
    drug_ids = le_drug.fit_transform(drug_names)
    cell_ids = le_cell.fit_transform(cell_names)

    # --------------------------------------------------------------
    # Standardize gene expression values (zero mean, unit variance)
    # --------------------------------------------------------------
    scaler_input  = StandardScaler()
    scaler_output = StandardScaler()
    gene_expr_input  = scaler_input.fit_transform(gene_expr_input)
    gene_expr_output = scaler_output.fit_transform(gene_expr_output)

    # --------------------------------------------------------------
    # Convert numpy arrays to PyTorch tensors and move to device
    # --------------------------------------------------------------
    X = torch.tensor(gene_expr_input, dtype=torch.float32).to(device)
    y = torch.tensor(gene_expr_output, dtype=torch.float32).to(device)
    drug = torch.tensor(drug_ids, dtype=torch.float32).unsqueeze(1).to(device)
    cell = torch.tensor(cell_ids, dtype=torch.float32).unsqueeze(1).to(device)

    # Return tensors and gene dimension
    return X, y, drug, cell, X.shape[1]

