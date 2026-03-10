import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

from seir_framework.model.seir import SEIRModel
from seir_framework.inference.likelihood import NegativeBinomialLikelihood
from seir_framework.inference.smc import ParticleFilter
from seir_framework.inference.pso import AdaptivePSO
from seir_framework.utils.diagnostics import Diagnostics
from seir_framework.utils.viz import plot_estimates, animate_results
from generate_synthetic_data import generate_data

# ================= CONFIGURATION =================
DATA_FILE = 'data/synthetic_outbreak.csv'
POPULATION_SIZE = 100000 
OUTPUT_DIR = 'output'
# =================================================

def run_demo():
    print("=== SEIR Framework Complete Demo Run ===")
    
    # 1. Generate Data
    if not os.path.exists(DATA_FILE):
        print("\n[1/4] Generating realistic synthetic data...")
        generate_data(DATA_FILE)
    else:
        print(f"\n[1/4] Using existing data at {DATA_FILE}")

    # Load Data
    df = pd.read_csv(DATA_FILE)
    observed_data = df['cases'].values
    observed_data = np.nan_to_num(observed_data)
    days = len(observed_data)
    print(f"Loaded {days} days of outbreak data.")

    # Setup Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Setup Model
    print("\n[2/4] Initializing SEIR Model & Likelihood...")
    # Initial guesses
    default_params = {'beta': 0.5, 'sigma': 0.2, 'gamma': 0.1}
    model = SEIRModel(N=POPULATION_SIZE, params=default_params)
    obs_model = NegativeBinomialLikelihood()

    # 3. Optimization (PSO)
    print("\n[3/4] Estimating initial parameters (PSO)...")
    param_bounds = {
        'beta': (0.1, 2.0),
        'sigma': (0.1, 0.5),
        'gamma': (0.05, 0.3),
        'rho': (0.2, 1.0)
    }
    
    # Heuristic initial state
    i0 = max(1, observed_data[0])
    e0 = i0
    s0 = POPULATION_SIZE - i0 - e0
    initial_state = np.array([s0, e0, i0, 0, 0])
    
    pso = AdaptivePSO(model, observed_data, obs_model, param_bounds, population_size=50)
    best_params = pso.optimize(initial_state, max_iter=50)
    print(f"PSO Best Estimates: {best_params}")

    # 4. Inference (SMC)
    print("\n[4/4] Running Particle Filter (SMC) with Animation...")
    pf = ParticleFilter(model, obs_model, n_particles=500, ess_threshold=0.5)
    
    def trunc_norm(mean, std, low, high, size=None):
        vals = np.random.normal(mean, std, size)
        return np.clip(vals, low, high)

    # Allow beta to vary significantly to capture the wave
    param_priors = {
        'beta': lambda size: trunc_norm(best_params['beta'], 0.2, 0.0, 3.0, size),
        'sigma': lambda size: trunc_norm(best_params['sigma'], 0.05, 0.05, 1.0, size),
        'gamma': lambda size: trunc_norm(best_params['gamma'], 0.05, 0.05, 1.0, size),
        'rho': lambda size: trunc_norm(best_params['rho'], 0.1, 0.1, 1.0, size),
        'kappa': lambda size: np.random.uniform(2.0, 20.0, size)
    }

    pf.initialize(initial_state_priors={'S': s0, 'E': e0, 'I': i0, 'R': 0, 'C': 0},
                  param_priors=param_priors)
    
    pf.set_parameter_walk('beta', sigma=0.05) 

    # Run loop with progress bar
    for t in tqdm(range(days), desc="SMC Progress"):
        obs = observed_data[t]
        pf.step(t, dt=1.0, observed_data=obs)

    # 5. Results
    print("\nGenerating outputs...")
    diag = Diagnostics(pf.get_posterior_estimates())
    
    # Static Plot
    fig = plot_estimates(diag, observed_data, title="Analysis Results")
    plot_path = os.path.join(OUTPUT_DIR, 'final_results.png')
    fig.savefig(plot_path)
    print(f"Static plot saved to {plot_path}")
    
    # Animation
    anim_path = os.path.join(OUTPUT_DIR, 'simulation.gif')
    print(f"Creating animation at {anim_path} (this may take a moment)...")
    animate_results(diag, observed_data, anim_path)
    print("Done!")

if __name__ == "__main__":
    run_demo()
