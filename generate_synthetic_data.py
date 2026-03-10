import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from seir_framework.model.seir import SEIRModel

def generate_data(output_path='data/synthetic_outbreak.csv', seed=None):
    print(f"Generating synthetic outbreak data (seed={seed})...")
    
    # 1. Setup True Parameters
    N = 100000
    true_params = {
        'beta': 0.6,    # High initial spread
        'sigma': 0.2,   # 5 day incubation
        'gamma': 0.1,   # 10 day infectious
    }
    
    # Time-varying beta: Intervention at day 40, Relaxation at day 80
    def beta_func(t):
        if t < 40: return 0.6          # Fast spread
        elif t < 80: return 0.2        # Lockdown / Intervention
        else: return 0.35              # Partial Reopening
        
    model = SEIRModel(N=N, params=true_params, seed=seed)
    
    # 2. Run Simulation
    t_max = 120
    initial_state = np.array([N-5, 0, 5, 0, 0]) # Start with 5 infected
    
    time_points, history = model.run(0, t_max, initial_state, 
                                     time_varying_params={'beta': beta_func}, 
                                     mode='stochastic')
    
    # 3. Create Observations
    # Incidence = C[t] - C[t-1]
    cumulative = history[:, 4]
    incidence = np.diff(cumulative)
    incidence = np.maximum(incidence, 0)
    
    # Add noise (Negative Binomial)
    rng = np.random.default_rng(seed)
    rho = 0.7       # 70% Reporting rate
    kappa = 10.0    # Dispersion
    
    obs_mean = rho * incidence
    # Avoid div by zero
    obs_mean = np.maximum(obs_mean, 1e-9)
    
    # p = kappa / (kappa + mu)
    p = kappa / (kappa + obs_mean)
    observed_cases = rng.negative_binomial(n=kappa, p=p)
    
    # 4. Save to CSV
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=len(observed_cases))
    
    df = pd.DataFrame({
        'date': dates,
        'cases': observed_cases
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    generate_data()
