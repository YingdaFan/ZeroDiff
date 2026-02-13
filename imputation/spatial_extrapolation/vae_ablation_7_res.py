#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import random

# =============================================================================
# 0. Data Preprocessing Configuration
# =============================================================================

DATA_PATH = '../../denormalized_camels_data_time.parquet'
TRAIN_DATES = ('1989-01-01', '2001-12-31')
TEST_DATES = ('2002-01-01', '2004-12-31')

METEO_FEATURES = ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
TARGET = 'QObs(mm/d)'
STATIC_FEATURES = [
    'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity',
    'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
    'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
    'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac',
    'clay_frac', 'elev_mean', 'slope_mean', 'area_gages2',
    'carbonate_rocks_frac', 'geol_permeability'
]

# =============================================================================
# 0.1 Data Preprocessing Functions
# =============================================================================

def load_and_filter_data(data_path, start_date, end_date):
    """Load data and filter by date range."""
    print(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    df['Time'] = pd.to_datetime(df['Time'])

    # Filter by date range
    mask = (df['Time'] >= start_date) & (df['Time'] <= end_date)
    df_filtered = df[mask].copy()

    print(f"  Original data: {len(df)} rows")
    print(f"  Filtered ({start_date} ~ {end_date}): {len(df_filtered)} rows")
    print(f"  Contains {df_filtered['basin_id'].nunique()} basins")

    return df_filtered


def compute_basin_statistics(df, meteo_features, target, static_features):
    """Compute statistics for each basin."""
    print("\nComputing statistics for each basin...")

    basins = df['basin_id'].unique()
    stats_list = []

    for basin in basins:
        basin_data = df[df['basin_id'] == basin]

        # Remove invalid QObs values (NaN and negative)
        valid_mask = (basin_data[target].notna()) & (basin_data[target] >= 0)
        basin_data_valid = basin_data[valid_mask]

        # Compute meteorological feature statistics
        meteo_mean = basin_data_valid[meteo_features].mean().values
        meteo_std = basin_data_valid[meteo_features].std().values

        # Compute flow statistics
        target_mean = basin_data_valid[target].mean()
        target_std = basin_data_valid[target].std()

        # Get static features (same for all rows, take first row)
        static = basin_data_valid[static_features].iloc[0].values

        # Combine all statistics
        stats = {
            'basin_id': basin,
            'meteo_mean': meteo_mean,
            'meteo_std': meteo_std,
            'static': static,
            'target_mean': target_mean,
            'target_std': target_std
        }
        stats_list.append(stats)

    print(f"  Successfully computed statistics for {len(stats_list)} basins")
    return stats_list


def build_feature_target_matrices(train_stats, test_stats):
    """Build feature matrix X and target matrix Y."""
    print("\nBuilding feature and target matrices...")

    def extract_matrices(stats_list):
        meteo_means = np.vstack([s['meteo_mean'] for s in stats_list])
        meteo_stds = np.vstack([s['meteo_std'] for s in stats_list])
        statics = np.vstack([s['static'] for s in stats_list])

        # X = [meteo_mean, meteo_std, static]
        X = np.hstack([meteo_means, meteo_stds, statics])

        # Y = [target_mean, target_std]
        Y = np.column_stack([
            [s['target_mean'] for s in stats_list],
            [s['target_std'] for s in stats_list]
        ])

        basin_ids = np.array([s['basin_id'] for s in stats_list])

        return X, Y, basin_ids

    X_train, Y_train, train_basins = extract_matrices(train_stats)
    X_test, Y_test, test_basins = extract_matrices(test_stats)

    print(f"  Training set: X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"  Test set: X_test {X_test.shape}, Y_test {Y_test.shape}")
    print(f"  Feature count: {X_train.shape[1]} = 5(meteo mean) + 5(meteo std) + {len(STATIC_FEATURES)}(static)")

    return X_train, Y_train, X_test, Y_test, train_basins, test_basins


# =============================================================================
# 1. VAE Model Architecture
# =============================================================================

class BasinFlowVAE(nn.Module):
    """
    Variational Autoencoder for learning basin flow distributions.

    Architecture:
    - Encoder: Basin features + flow sequence -> latent distribution parameters (mu, sigma)
    - Decoder: Basin features + latent code -> flow distribution parameters
    """

    def __init__(self,
                 feature_dim: int = 37,      # Meteorological statistics + static attributes
                 flow_dim: int = 365,         # Flow sequence length (or statistics dimension)
                 latent_dim: int = 32,        # Latent space dimension
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()

        self.feature_dim = feature_dim
        self.flow_dim = flow_dim
        self.latent_dim = latent_dim

        # ========== Encoder ==========
        # Input: basin features + flow data
        encoder_input_dim = feature_dim + flow_dim

        encoder_layers = []
        in_dim = encoder_input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # ========== Decoder ==========
        # Input: basin features + latent code
        decoder_input_dim = feature_dim + latent_dim

        decoder_layers = []
        in_dim = decoder_input_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Output flow distribution parameters
        self.fc_flow_mu = nn.Linear(hidden_dims[0], flow_dim)
        self.fc_flow_logvar = nn.Linear(hidden_dims[0], flow_dim)

    def encode(self, features: torch.Tensor, flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder: encode basin features and flow into latent distribution."""
        x = torch.cat([features, flows], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decoder: generate flow distribution from latent code."""
        x = torch.cat([z, features], dim=1)
        h = self.decoder(x)
        flow_mu = self.fc_flow_mu(h)
        flow_logvar = self.fc_flow_logvar(h)
        return flow_mu, flow_logvar

    def forward(self, features: torch.Tensor, flows: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode
        z_mu, z_logvar = self.encode(features, flows)
        z = self.reparameterize(z_mu, z_logvar)

        # Decode
        flow_mu, flow_logvar = self.decode(z, features)

        return {
            'flow_mu': flow_mu,
            'flow_logvar': flow_logvar,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'z': z
        }

    def generate(self, features: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Generate flow samples for a new basin."""
        self.eval()
        with torch.no_grad():
            batch_size = features.shape[0]
            samples = []

            for _ in range(n_samples):
                # Sample latent variables from standard normal distribution
                z = torch.randn(batch_size, self.latent_dim).to(features.device)

                # Decode to generate flow
                flow_mu, flow_logvar = self.decode(z, features)
                flow_std = torch.exp(0.5 * flow_logvar)

                # Sample flow values
                flow_sample = flow_mu + torch.randn_like(flow_mu) * flow_std
                samples.append(flow_sample.unsqueeze(1))

            return torch.cat(samples, dim=1)  # [batch, n_samples, flow_dim]


# =============================================================================
# 2. Conditional VAE Variant (Recommended)
# =============================================================================

class ConditionalBasinVAE(nn.Module):
    """
    Conditional VAE: more suitable for missing data scenarios + residual connection.
    Latent variable z captures shared patterns across basins,
    conditional variable c is the basin features.

    Improvement: adds residual connection to enhance feature learning capability.
    """

    def __init__(self,
                 feature_dim: int = 37,
                 latent_dim: int = 16,
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        # Encoder: encode flow into latent space only
        self.flow_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Flow mean and std
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.z_mu = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

        # Feature encoder with Residual Connection
        self.feature_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.feature_relu1 = nn.ReLU()
        self.feature_dropout1 = nn.Dropout(dropout)
        self.feature_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.feature_relu2 = nn.ReLU()
        self.feature_dropout2 = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Output flow mean and std
        )

    def encode(self, flow_stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode flow statistics y into latent distribution z."""
        h = self.flow_encoder(flow_stats)
        z_mu = self.z_mu(h)
        z_logvar = self.z_logvar(h)
        return z_mu, z_logvar

    def decode(self, z: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Decode: generate flow statistics y from latent code z and basin features x.
        Uses residual connection and dropout regularization.
        """
        # Feature encoder with residual connection
        h1 = self.feature_fc1(features)
        h1 = self.feature_relu1(h1)
        h1 = self.feature_dropout1(h1)
        h2 = self.feature_fc2(h1)
        feature_h = self.feature_relu2(h2 + h1)  # Residual connection
        feature_h = self.feature_dropout2(feature_h)
        combined = torch.cat([z, feature_h], dim=1)
        flow_stats = self.decoder(combined)
        return flow_stats

    def forward(self, features: torch.Tensor, flow_stats: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        z_mu, z_logvar = self.encode(flow_stats)

        # Reparameterize
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + eps * std

        # Decode
        flow_pred = self.decode(z, features)

        return {
            'flow_pred': flow_pred,
            'z_mu': z_mu,
            'z_logvar': z_logvar
        }

    def predict_new_basin(self, features: torch.Tensor, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Predict flow distribution for a new basin."""
        self.eval()
        with torch.no_grad():
            samples = []

            for _ in range(n_samples):
                # Sample z from prior distribution z ~ N(0, I)
                z = torch.randn(features.shape[0], self.z_mu.out_features).to(features.device)

                # Decode to generate flow statistics
                flow_stats = self.decode(z, features)
                samples.append(flow_stats)

            samples = torch.stack(samples, dim=1)  # [batch, n_samples, 2]

            # Compute statistics
            mean_estimate = samples.mean(dim=1).cpu().numpy()
            std_estimate = samples.std(dim=1).cpu().numpy()
            percentiles = np.percentile(samples.cpu().numpy(), [5, 25, 50, 75, 95], axis=1)

            return {
                'mean': mean_estimate,
                'std': std_estimate,
                'percentiles': percentiles,
                'samples': samples.cpu().numpy()
            }


# =============================================================================
# 3. Loss Function
# =============================================================================

def vae_loss(outputs: Dict[str, torch.Tensor],
             targets: torch.Tensor,
             beta: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    VAE loss = Reconstruction loss + beta * KL divergence

    Parameters:
    - beta: weight for KL term (beta-VAE)
    """

    # Reconstruction loss (negative log-likelihood)
    flow_mu = outputs['flow_mu'] if 'flow_mu' in outputs else outputs['flow_pred']
    recon_loss = F.mse_loss(flow_mu, targets, reduction='mean')

    # KL divergence
    z_mu = outputs['z_mu']
    z_logvar = outputs['z_logvar']
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()) / z_mu.shape[0]

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss
    }


# =============================================================================
# 4. Dataset Class
# =============================================================================

class BasinDataset(Dataset):
    """Basin dataset."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, basin_ids: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.basin_ids = basin_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.basin_ids[idx]


# =============================================================================
# 5. Training Function
# =============================================================================

def train_vae(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              lr: float = 1e-3,
              beta_schedule: str = 'linear',
              beta_value: float = 0.1,
              random_seed: int = 42) -> Dict[str, List[float]]:
    """
    Train the VAE model.

    Parameters:
    -----------
    random_seed : int
        Random seed for reproducibility (default 42).
    """

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'train_recon': [],
               'train_kl': [], 'val_recon': [], 'val_kl': []}

    for epoch in range(epochs):
        # Beta scheduling for beta-VAE
        if beta_schedule == 'linear':
            # Linear warmup from 0 to beta_value
            beta = min(beta_value, (epoch / (epochs // 2)) * beta_value)
        else:
            # Constant beta
            beta = beta_value

        # Training
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}

        for features, flow_stats, _ in train_loader:
            features = features.to(device)
            flow_stats = flow_stats.to(device)

            optimizer.zero_grad()
            outputs = model(features, flow_stats)
            losses = vae_loss(outputs, flow_stats, beta=beta)

            losses['total'].backward()
            optimizer.step()

            for key in train_losses:
                train_losses[key] += losses[key].item()

        # Validation
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}

        with torch.no_grad():
            for features, flow_stats, _ in val_loader:
                features = features.to(device)
                flow_stats = flow_stats.to(device)

                outputs = model(features, flow_stats)
                losses = vae_loss(outputs, flow_stats, beta=beta)

                for key in val_losses:
                    val_losses[key] += losses[key].item()

        # Record history
        n_train = len(train_loader)
        n_val = len(val_loader)

        history['train_loss'].append(train_losses['total'] / n_train)
        history['train_recon'].append(train_losses['recon'] / n_train)
        history['train_kl'].append(train_losses['kl'] / n_train)
        history['val_loss'].append(val_losses['total'] / n_val)
        history['val_recon'].append(val_losses['recon'] / n_val)
        history['val_kl'].append(val_losses['kl'] / n_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {history['train_loss'][-1]:.4f} "
                  f"(Recon: {history['train_recon'][-1]:.4f}, KL: {history['train_kl'][-1]:.4f})")
            print(f"  Val Loss: {history['val_loss'][-1]:.4f} "
                  f"(Recon: {history['val_recon'][-1]:.4f}, KL: {history['val_kl'][-1]:.4f})")

    return history


# =============================================================================
# 6. Application Example
# =============================================================================

def apply_vae_to_basin_prediction():
    """
    Use VAE to predict the flow distribution of the 531st basin.
    """

    # 1. Load and preprocess from raw data

    # Load training and test data
    df_train = load_and_filter_data(DATA_PATH, TRAIN_DATES[0], TRAIN_DATES[1])
    df_test = load_and_filter_data(DATA_PATH, TEST_DATES[0], TEST_DATES[1])

    # Compute statistics
    train_stats = compute_basin_statistics(df_train, METEO_FEATURES, TARGET, STATIC_FEATURES)
    test_stats = compute_basin_statistics(df_test, METEO_FEATURES, TARGET, STATIC_FEATURES)

    # Build feature and target matrices
    X_train, Y_train, X_test, Y_test, train_basins, test_basins = \
        build_feature_target_matrices(train_stats, test_stats)

    print(f"\nData overview:")
    print(f"  Training basins: {X_train.shape[0]}")
    print(f"  Test basins: {X_test.shape[0]}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Y range: mean={Y_train[:, 0].min():.3f}-{Y_train[:, 0].max():.3f}, "
          f"std={Y_train[:, 1].min():.3f}-{Y_train[:, 1].max():.3f}")

    # 2. Spatial extrapolation setup: hold out the 531st basin as "new basin"
    print("\n2. Spatial extrapolation setup (Leave-One-Out)...")

    # Training set: first 530 basins
    X_train_530 = X_train[:530]
    Y_train_530 = Y_train[:530]
    train_basins_530 = train_basins[:530]

    # The 531st basin (never seen "new basin")
    X_new_basin = X_train[530:531]  # Features of the 531st basin
    Y_new_basin = Y_train[530:531]  # True values of the 531st basin (for comparison)
    new_basin_id = train_basins[530]

    print(f"  Training set: first 530 basins")
    print(f"  New basin: Basin {new_basin_id} (531st, never seen during training)")
    print(f"  New basin true values: mean={Y_new_basin[0, 0]:.3f}, std={Y_new_basin[0, 1]:.3f}")

    # 3. Normalize X and Y features (based on 530 basins only)
    print("\n3. Normalizing features (based on 530 basins)...")

    # Important: normalization parameters computed from 530 basins only
    X_mean = X_train_530.mean(axis=0)
    X_std = X_train_530.std(axis=0) + 1e-8
    Y_mean = Y_train_530.mean(axis=0)
    Y_std = Y_train_530.std(axis=0) + 1e-8

    # Normalize training set (530 basins)
    X_train_530_normalized = (X_train_530 - X_mean) / X_std
    Y_train_530_normalized = (Y_train_530 - Y_mean) / Y_std

    # Normalize new basin (using 530 basins statistics)
    X_new_basin_normalized = (X_new_basin - X_mean) / X_std

    print(f"  Normalized X range: [{X_train_530_normalized.min():.3f}, {X_train_530_normalized.max():.3f}]")
    print(f"  Normalized Y range: [{Y_train_530_normalized.min():.3f}, {Y_train_530_normalized.max():.3f}]")
    print(f"  Y normalization parameters: mean={Y_mean}, std={Y_std}")

    # 4. Create data loaders (train with all 530 basins)
    train_dataset = BasinDataset(X_train_530_normalized, Y_train_530_normalized, train_basins_530)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(f"\n  Data split:")
    print(f"    Training: 530 basins (all used for training)")
    print(f"    New basin test: 1 basin (Basin {new_basin_id})")

    # 5. Initialize model
    print("\n4. Initializing ConditionalBasinVAE model (Ablation 7: ReLU + Residual Connection)...")
    print("   Architecture improvements:")
    print("   - Feature encoder ReLU activation")
    print("   - Feature encoder second layer residual connection")
    model = ConditionalBasinVAE(feature_dim=X_train.shape[1])

    # 6. Train model (using all 530 basins)
    print("\n5. Training model...")
    print(f"  Using all 530 basins for training")
    print(f"  The 531st basin (Basin {new_basin_id}) is completely excluded from training!")
    # Validation set also uses training set (no separate validation needed)
    history = train_vae(model, train_loader, train_loader, epochs=100)

    # 7. Predict the 531st "new basin"
    print("\n6. Predicting new basin (never seen during training)...")
    print(f"  Target: Basin {new_basin_id}")

    # Use normalized new basin features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_basin_features = torch.FloatTensor(X_new_basin_normalized).to(device)

    predictions_normalized = model.predict_new_basin(new_basin_features, n_samples=1000)

    # Denormalize: convert predictions from normalized space back to original space
    print("\n7. Denormalizing prediction results...")
    predictions = {
        'mean': predictions_normalized['mean'] * Y_std + Y_mean,
        'std': predictions_normalized['std'] * Y_std,  # Std only multiplied by scale factor
        'percentiles': predictions_normalized['percentiles'] * Y_std[np.newaxis, np.newaxis, :] + Y_mean[np.newaxis, np.newaxis, :],
        'samples': predictions_normalized['samples'] * Y_std[np.newaxis, np.newaxis, :] + Y_mean[np.newaxis, np.newaxis, :]
    }

    print(f"\nPrediction results (original scale):")
    print(f"  Flow mean: {predictions['mean'][0, 0]:.3f} +/- {predictions['std'][0, 0]:.3f} mm/day")
    print(f"  Flow std: {predictions['mean'][0, 1]:.3f} +/- {predictions['std'][0, 1]:.3f} mm/day")
    print(f"  95% confidence interval:")
    print(f"    Mean: [{predictions['percentiles'][0, 0, 0]:.3f}, {predictions['percentiles'][4, 0, 0]:.3f}]")
    print(f"    Std: [{predictions['percentiles'][0, 0, 1]:.3f}, {predictions['percentiles'][4, 0, 1]:.3f}]")

    print(f"\nTrue values (Basin {new_basin_id}):")
    print(f"  Flow mean: {Y_new_basin[0, 0]:.3f} mm/day")
    print(f"  Flow std: {Y_new_basin[0, 1]:.3f} mm/day")

    # Compute prediction error
    mean_error = abs(predictions['mean'][0, 0] - Y_new_basin[0, 0]) / Y_new_basin[0, 0] * 100
    std_error = abs(predictions['mean'][0, 1] - Y_new_basin[0, 1]) / Y_new_basin[0, 1] * 100

    print(f"\nPrediction error:")
    print(f"  Flow mean error: {mean_error:.2f}%")
    print(f"  Flow std error: {std_error:.2f}%")

    # Check if true values fall within confidence interval
    mean_in_ci = predictions['percentiles'][0, 0, 0] <= Y_new_basin[0, 0] <= predictions['percentiles'][4, 0, 0]
    std_in_ci = predictions['percentiles'][0, 0, 1] <= Y_new_basin[0, 1] <= predictions['percentiles'][4, 0, 1]

    print(f"\nConfidence interval check:")
    print(f"  Mean within 95% CI: {'Yes' if mean_in_ci else 'No'}")
    print(f"  Std within 95% CI: {'Yes' if std_in_ci else 'No'}")

    # 8. Visualize prediction distribution
    visualize_predictions(predictions, Y_new_basin[0])

    return model, predictions


def visualize_predictions(predictions: Dict[str, np.ndarray],
                         true_values: np.ndarray = None):
    """Visualize the VAE predicted distributions."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution of flow mean
    ax = axes[0]
    samples_mean = predictions['samples'][0, :, 0]
    ax.hist(samples_mean, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(predictions['mean'][0, 0], color='red', linestyle='--',
               label=f'Mean: {predictions["mean"][0, 0]:.3f}')
    if true_values is not None:
        ax.axvline(true_values[0], color='green', linestyle='-',
                   label=f'True: {true_values[0]:.3f}')
    ax.set_xlabel('Flow Mean (mm/day)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Predicted Flow Mean Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distribution of flow std
    ax = axes[1]
    samples_std = predictions['samples'][0, :, 1]
    ax.hist(samples_std, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(predictions['mean'][0, 1], color='red', linestyle='--',
               label=f'Mean: {predictions["mean"][0, 1]:.3f}')
    if true_values is not None:
        ax.axvline(true_values[1], color='green', linestyle='-',
                   label=f'True: {true_values[1]:.3f}')
    ax.set_xlabel('Flow Std (mm/day)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Predicted Flow Std Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('VAE Predicted Distribution vs True Values', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7. Advanced: Basin Similarity Analysis
# =============================================================================

def analyze_basin_similarity(model: nn.Module,
                            X: np.ndarray,
                            Y: np.ndarray,
                            basin_ids: np.ndarray):
    """
    Analyze basin similarity using the VAE latent space.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Encode all basins into latent space
        features = torch.FloatTensor(X).to(device)
        flow_stats = torch.FloatTensor(Y).to(device)

        if isinstance(model, ConditionalBasinVAE):
            z_mu, _ = model.encode(flow_stats)
        else:
            outputs = model(features, flow_stats)
            z_mu = outputs['z_mu']

        latent_embeddings = z_mu.cpu().numpy()

    # Dimensionality reduction visualization (using t-SNE or UMAP)
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(latent_embeddings)

    # Visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=Y[:, 0], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Flow Mean (mm/day)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Basin Similarity in VAE Latent Space')
    plt.grid(True, alpha=0.3)

    # Add some basin ID labels
    for i in range(0, len(basin_ids), len(basin_ids)//10):
        plt.annotate(basin_ids[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.show()

    return latent_embeddings, embeddings_2d


if __name__ == "__main__":

    # Run experiment
    model, predictions = apply_vae_to_basin_prediction()
