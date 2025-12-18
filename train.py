import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

from metrics import evaluate


def train_gan(G_class, D_class, X, y, drug, cell, gene_dim, device, name):
    """
    Train a conditional GAN model using K-fold cross-validation.

    The generator is optimized with a combination of reconstruction loss
    (MSE) and adversarial loss, while the discriminator is trained to
    distinguish real and generated gene expression profiles conditioned
    on drug and cell line information.

    Parameters
    ----------
    G_class : torch.nn.Module
        Generator class (e.g., ResidualGenerator).
    D_class : torch.nn.Module
        Discriminator class (e.g., MLPDiscriminator).
    X : torch.Tensor
        Input gene expression tensor of shape (N, G).
    y : torch.Tensor
        Target gene expression tensor of shape (N, G).
    drug : torch.Tensor
        Encoded drug labels of shape (N, 1).
    cell : torch.Tensor
        Encoded cell line labels of shape (N, 1).
    gene_dim : int
        Number of genes (feature dimension).
    device : torch.device
        Device used for training (CPU or CUDA).
    name : str
        Model name identifier.

    Returns
    -------
    name : str
        Model name.
    metrics_mean : numpy.ndarray
        Mean values of generator evaluation metrics across folds.
    metrics_std : numpy.ndarray
        Standard deviations of generator evaluation metrics across folds.
    d_metrics_mean : numpy.ndarray
        Mean discriminator performance metrics (accuracy and AUC).
    """

    # --------------------------------------------------------------
    # K-fold cross-validation setup
    # --------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics_all = []
    d_metrics_all = []

    # --------------------------------------------------------------
    # Cross-validation loop
    # --------------------------------------------------------------
    for train_idx, test_idx in kf.split(X):

        # Split data into training and test sets
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        d_tr, d_te = drug[train_idx], drug[test_idx]
        c_tr, c_te = cell[train_idx], cell[test_idx]

        # Initialize generator and discriminator
        G = G_class(gene_dim).to(device)
        D = D_class(gene_dim).to(device)

        # Optimizers
        opt_G = optim.Adam(G.parameters(), lr=1e-3)
        opt_D = optim.Adam(D.parameters(), lr=1e-3)

        # Loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        # ----------------------------------------------------------
        # Training loop
        # ----------------------------------------------------------
        for _ in range(20):  # number of epochs
            for i in range(0, len(X_tr), 32):  # mini-batch training

                xb = X_tr[i:i + 32]
                yb = y_tr[i:i + 32]
                db = d_tr[i:i + 32]
                cb = c_tr[i:i + 32]

                # ----------------------
                # Train Discriminator
                # ----------------------
                opt_D.zero_grad()

                # Real samples
                real_logits = D(yb, db, cb)

                # Fake samples (detach to avoid updating generator)
                fake_logits = D(G(xb, db, cb).detach(), db, cb)

                # Discriminator loss
                d_loss = (
                    bce_loss(real_logits, torch.ones_like(real_logits)) +
                    bce_loss(fake_logits, torch.zeros_like(fake_logits))
                )

                d_loss.backward()
                opt_D.step()

                # ----------------------
                # Train Generator
                # ----------------------
                opt_G.zero_grad()

                # Generate predictions
                gen = G(xb, db, cb)
                adv_logits = D(gen, db, cb)

                # Generator loss: reconstruction + adversarial
                g_loss = (
                    mse_loss(gen, yb) +
                    0.1 * bce_loss(adv_logits, torch.ones_like(adv_logits))
                )

                g_loss.backward()
                opt_G.step()

        # ----------------------------------------------------------
        # Evaluation on test set
        # ----------------------------------------------------------
        with torch.no_grad():
            y_pred = G(X_te, d_te, c_te).cpu().numpy()
            y_true = y_te.cpu().numpy()

            # Generator performance metrics
            metrics_all.append(evaluate(y_true, y_pred))

            # Discriminator performance metrics
            real_scores = D(y_te, d_te, c_te).cpu().numpy().ravel()
            fake_scores = D(
                torch.tensor(y_pred, dtype=torch.float32).to(device),
                d_te,
                c_te
            ).cpu().numpy().ravel()

            labels = np.concatenate([
                np.ones_like(real_scores),
                np.zeros_like(fake_scores)
            ])
            scores = np.concatenate([real_scores, fake_scores])

            d_metrics_all.append([
                accuracy_score(labels, scores > 0),
                roc_auc_score(labels, scores)
            ])

    # --------------------------------------------------------------
    # Aggregate results across folds
    # --------------------------------------------------------------
    return (
        name,
        np.mean(metrics_all, axis=0),
        np.std(metrics_all, axis=0),
        np.mean(d_metrics_all, axis=0)
    )
