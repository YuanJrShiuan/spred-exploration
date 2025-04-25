import matplotlib.pyplot as plt
import numpy as np
import torch
from data import load_darwin, load_advertise
from routines import run_lasso, run_rs_regression
import pandas as pd

def export_weights(weights, filename):
    weights = to_numpy(weights).flatten()
    df = pd.DataFrame({'weight': weights})
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

def to_numpy(weights):
    if isinstance(weights, torch.Tensor):
        return weights.detach().cpu().numpy()
    return weights

def normalize_weights(weights):
    weights = to_numpy(weights)
    return weights / np.max(np.abs(weights))

def get_weights(dataset_loader, alpha=0.01, method='lasso', device='cuda:0'):
    (x, y), _ = dataset_loader()
    if method in ['lasso', 'LARS']:
        result = run_lasso(alpha, x, y, method=method)
    else:
        result = run_rs_regression(alpha, x, y, device=device, loss_func='mse', epochs=1000, eval_every_epoch=200)
    return result['weights'].flatten()

# Get weights

darwin_lasso_weights = normalize_weights(get_weights(load_darwin, alpha=0.07704, method='lasso'))
darwin_spred_weights = normalize_weights(get_weights(load_darwin, alpha=0.261127, method='spred'))

advertise_lasso_weights = normalize_weights(get_weights(load_advertise, alpha=0.006705, method='lasso'))
advertise_spred_weights = normalize_weights(get_weights(load_advertise, alpha=0.07704, method='spred'))

# Create histogram plots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# Plot 1: DARWIN
axs[0].hist(np.abs(darwin_lasso_weights), bins=50, alpha=0.7, label='Spred Lasso', color='coral', log=True)
axs[0].hist(np.abs(darwin_spred_weights), bins=50, alpha=0.7, label='Lasso', color='skyblue', log=True)
axs[0].set_title('DARWIN Dataset - |Weight| Histogram')
axs[0].set_xlabel('|Weight Magnitude|')
axs[0].set_ylabel('Log Count')
axs[0].legend()

# Plot 2: Advertisement
axs[1].hist(np.abs(advertise_lasso_weights), bins=50, alpha=0.7, label='Spred Lasso', color='coral', log=True)
axs[1].hist(np.abs(advertise_spred_weights), bins=50, alpha=0.7, label='Lasso', color='skyblue', log=True)
axs[1].set_title('Advertisement Dataset - |Weight| Histogram')
axs[1].set_xlabel('|Weight Magnitude|')
axs[1].set_ylabel('Log Count')
axs[1].legend()

# Global title and display
plt.suptitle('Histogram of Absolute Weight Coefficients (Lasso vs Spred)', fontsize=14)
plt.savefig('weight_histograms.png', dpi=300)
plt.show()

export_weights(darwin_lasso_weights, 'darwin_lasso_weights.csv')
export_weights(darwin_spred_weights, 'darwin_spred_weights.csv')
export_weights(advertise_lasso_weights, 'advertise_lasso_weights.csv')
export_weights(advertise_spred_weights, 'advertise_spred_weights.csv')

