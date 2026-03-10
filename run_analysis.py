import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

from seir_framework.model.seir import SEIRModel
from seir_framework.inference.likelihood import NegativeBinomialLikelihood
from seir_framework.inference.smc import ParticleFilter
from seir_framework.inference.pso import AdaptivePSO
from seir_framework.utils.diagnostics import Diagnostics
from seir_framework.utils.viz import plot_estimates

# ================= CONFIGURATION =================
# Path to your CSV file. It must have a column named 'cases'.
DATA_FILE = 'data/example_cases.csv' 

# Total population size of the region
POPULATION_SIZE = 100000 

# Output directory
OUTPUT_DIR = 'output'
# =================================================

def run_analysis(data_path, population_size):
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    if 'cases' not in df.columns:
        print("Error: CSV must have a 'cases' column.")
        return
        
    observed_data = df['cases'].values
    # Handle NaNs if any (fill with 0 or interp? simple drop or fill 0 for now)
    observed_data = np.nan_to_num(observed_data)
    
    days = len(observed_data)
    print(f"Loaded {days} days of data.")

    # Setup Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Setup Model
    print("\nInitializing SEIR Model...")
    # Initial guesses for params
    default_params = {
        'beta': 0.5,
        'sigma': 0.2, # ~5 days incubation
        'gamma': 0.1, # ~10 days infectious
    }
    model = SEIRModel(N=population_size, params=default_params)
    obs_model = NegativeBinomialLikelihood()

    # 2. PSO for Initialization
    print("\nRunning PSO to estimate initial parameters...")
    param_bounds = {
        'beta': (0.05, 2.0),
        'sigma': (0.1, 0.5), # Constrain to biological plausibility
        'gamma': (0.05, 0.3), # Constrain to biological plausibility
        'rho': (0.1, 1.0)     # Reporting rate
    }
    
    # Heuristic initial state
    # Assume first data point represents initial infections roughly
    i0 = max(1, observed_data[0])
    e0 = i0
    s0 = population_size - i0 - e0
    initial_state = np.array([s0, e0, i0, 0, 0])
    
    pso = AdaptivePSO(model, observed_data, obs_model, param_bounds, population_size=40)
    best_params = pso.optimize(initial_state, max_iter=40)
    print(f"PSO Best Estimates: {best_params}")

    # 3. SMC / Particle Filter
    print("\nRunning Particle Filter (SMC) for time-varying inference...")
    pf = ParticleFilter(model, obs_model, n_particles=500, ess_threshold=0.5)
    
    # Define priors centered on PSO results
    def trunc_norm(mean, std, low, high, size=None):
        vals = np.random.normal(mean, std, size)
        return np.clip(vals, low, high)

    # Allow beta to vary (random walk), others static but uncertain
    param_priors = {
        'beta': lambda size: trunc_norm(best_params['beta'], 0.2, 0.0, 3.0, size),
        'sigma': lambda size: trunc_norm(best_params['sigma'], 0.05, 0.05, 1.0, size),
        'gamma': lambda size: trunc_norm(best_params['gamma'], 0.05, 0.05, 1.0, size),
        'rho': lambda size: trunc_norm(best_params['rho'], 0.1, 0.1, 1.0, size),
        'kappa': lambda size: np.random.uniform(2.0, 20.0, size)
    }

    pf.initialize(initial_state_priors={'S': s0, 'E': e0, 'I': i0, 'R': 0, 'C': 0},
                  param_priors=param_priors)
    
    # Enable random walk for beta (time-varying transmission)
    pf.set_parameter_walk('beta', sigma=0.05) 

    # Run loop
    for t in range(days):
        if t % 10 == 0:
            print(f"  Step {t}/{days}")
        
        obs = observed_data[t]
        pf.step(t, dt=1.0, observed_data=obs)

    # 4. Diagnostics & Plotting
    print("\nAnalyzing results...")
    diag = Diagnostics(pf.get_posterior_estimates())
    
    fig = plot_estimates(diag, observed_data, title="Analysis Results")
    plot_path = os.path.join(OUTPUT_DIR, 'analysis_plot.png')
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Print final stats
    beta_qs = diag.get_parameter_quantiles('beta')[-1]
    print(f"Final Estimated Beta: {beta_qs[1]:.3f} (95% CI: {beta_qs[0]:.3f} - {beta_qs[2]:.3f})")

if __name__ == "__main__":
    # You can also use command line args if preferred
    run_analysis(DATA_FILE, POPULATION_SIZE)
