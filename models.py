import torch
import torch.nn as nn


class ResidualGenerator(nn.Module):
    """
    Residual Generator network for conditional gene expression prediction.

    The generator takes as input:
        - baseline gene expression (x),
        - drug identity,
        - cell line identity,

    and predicts the residual change in gene expression induced by
    drug treatment. The final output is obtained via a residual
    connection: output = x + f(x, drug, cell).
    """

    def __init__(self, gene_dim, hidden_dim=256):
        """
        Parameters
        ----------
        gene_dim : int
            Number of genes (input and output feature dimension).
        hidden_dim : int, optional
            Number of hidden units in each fully connected layer.
        """
        super().__init__()

        # Fully connected layers
        # Input dimension: gene_dim + 2 (gene expression + drug + cell)
        self.fc1 = nn.Linear(gene_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, gene_dim)

        # Activation function
        self.act = nn.ReLU()

    def forward(self, x, drug, cell):
        """
        Forward pass of the generator.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression tensor of shape (N, gene_dim).
        drug : torch.Tensor
            Encoded drug labels of shape (N, 1).
        cell : torch.Tensor
            Encoded cell line labels of shape (N, 1).

        Returns
        -------
        torch.Tensor
            Predicted gene expression after drug perturbation.
        """

        # Concatenate gene expression with conditional information
        inp = torch.cat([x, drug, cell], dim=1)

        # Hidden layers
        h = self.act(self.fc1(inp))
        h = self.act(self.fc2(h))

        # Residual prediction
        return x + self.fc3(h)


class MLPDiscriminator(nn.Module):
    """
    Multi-layer perceptron (MLP) discriminator for conditional GAN training.

    The discriminator distinguishes real and generated gene expression
    profiles conditioned on drug and cell line information.
    """

    def __init__(self, gene_dim, hidden_dim=128):
        """
        Parameters
        ----------
        gene_dim : int
            Number of genes (feature dimension).
        hidden_dim : int, optional
            Number of hidden units in the first hidden layer.
        """
        super().__init__()

        # Sequential MLP architecture
        self.model = nn.Sequential(
            nn.Linear(gene_dim + 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, drug, cell):
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Gene expression tensor of shape (N, gene_dim).
        drug : torch.Tensor
            Encoded drug labels of shape (N, 1).
        cell : torch.Tensor
            Encoded cell line labels of shape (N, 1).

        Returns
        -------
        torch.Tensor
            Discriminator logits indicating real/fake predictions.
        """

        # Concatenate gene expression with conditional information
        inp = torch.cat([x, drug, cell], dim=1)

        return self.model(inp)
