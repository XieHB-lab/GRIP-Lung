import csv
from pathlib import Path
import torch

# Import utility functions and model definitions
from data_utils import load_data
from models import ResidualGenerator, MLPDiscriminator
from train import train_gan

# ------------------------------------------------------------------
# Device configuration: use GPU if available, otherwise fall back to CPU
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Define data directory and input file paths
# The actual data files should be placed under the "data/" folder
# ------------------------------------------------------------------
DATA_DIR = Path("data")

input_path  = DATA_DIR / "input_expression.csv"
output_path = DATA_DIR / "output_expression.csv"

# ------------------------------------------------------------------
# Load and preprocess data
# Returns:
#   X        : input gene expression tensor
#   y        : target gene expression tensor
#   drug     : encoded drug labels
#   cell     : encoded cell line labels
#   gene_dim : number of genes (feature dimension)
# ------------------------------------------------------------------
X, y, drug, cell, gene_dim = load_data(
    input_path,
    output_path,
    device
)

# ------------------------------------------------------------------
# Model configuration
# ResidualGenerator + MLPDiscriminator (GAN framework)
# ------------------------------------------------------------------
name = "Res+MLP"

# Train GAN model and evaluate performance using cross-validation
result = train_gan(
    ResidualGenerator,
    MLPDiscriminator,
    X, y, drug, cell,
    gene_dim,
    device,
    name
)

# Unpack results
# means   : mean values of evaluation metrics
# stds    : standard deviations of evaluation metrics
# d_means : discriminator performance metrics
names, means, stds, d_means = result

# ------------------------------------------------------------------
# Save results to CSV file
# ------------------------------------------------------------------
with open("res_mlp_result.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow(
        ["Model"]
        + list(means)
        + list(stds)
        + list(d_means)
    )

    # Write model results
    writer.writerow(
        [name]
        + list(means)
        + list(stds)
        + list(d_means)
    )

