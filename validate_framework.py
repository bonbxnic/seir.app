import numpy as np
import matplotlib.pyplot as plt
import os

from seir_framework.model.seir import SEIRModel
from seir_framework.inference.likelihood import NegativeBinomialLikelihood
from seir_framework.inference.smc import ParticleFilter
from seir_framework.inference.pso import AdaptivePSO
from seir_framework.utils.diagnostics import Diagnostics
from seir_framework.utils.viz import plot_estimates

def run_validation():
    print("=== SEIR Framework Validation Run ===")
    
    # 1. Generate Synthetic Data
    print("\n1. Generating Synthetic Data...")
    N = 100000
    true_params = {
        'beta': 0.4, # Will vary this over time
        'sigma': 0.2, # 5 days incubation
        'gamma': 0.1, # 10 days infectious
    }
    
    model = SEIRModel(N=N, params=true_params, seed=42)
    
    # Define time-varying beta
    def beta_func(t):
        if t < 30: return 0.4
        elif t < 60: return 0.2 # Intervention
        else: return 0.3 # Reopening
        
    t_max = 90
    initial_state = np.array([N-10, 0, 10, 0, 0]) # Start with 10 infected
    
    time_points, history = model.run(0, t_max, initial_state, 
                                     time_varying_params={'beta': beta_func}, 
                                     mode='stochastic')
    
    # Extract incidence (C[t] - C[t-1])
    # History has t=0..90 (91 points). Incidence will be for t=1..90
    cumulative_cases = history[:, 4]
    true_incidence = np.diff(cumulative_cases)
    
    # Add observation noise (NegBin)
    rng = np.random.default_rng(42)
    rho = 0.8 # Reporting rate
    kappa = 10.0 # Dispersion
    
    obs_mean = rho * true_incidence
    p = kappa / (kappa + obs_mean)
    # Ensure p is within (0, 1)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    
    observed_data = rng.negative_binomial(n=kappa, p=p)
    
    print(f"Generated {len(observed_data)} days of data.")
    
    # 2. PSO Initialization
    print("\n2. Running PSO for Initialization...")
    obs_model = NegativeBinomialLikelihood()
    
    # We don't know the true parameters. Let's guess bounds.
    param_bounds = {
        'beta': (0.1, 1.0),
        'sigma': (0.1, 0.5),
        'gamma': (0.05, 0.2),
        'rho': (0.4, 1.0) # Assume we suspect under-reporting
    }
    
    # NOTE: PSO uses deterministic model, which might differ from stochastic realization.
    pso = AdaptivePSO(model, observed_data, obs_model, param_bounds, population_size=30)
    
    # Initial guess for state? 
    # Usually we estimate initial I, but here we assume we know roughly start state or estimate it.
    # For simplicity, fix initial state to truth, or close to it.
    # In a real scenario, we'd estimate I0 too.
    pso_initial_state = np.array([N-10, 0, 10, 0, 0]) 
    
    best_params = pso.optimize(pso_initial_state, max_iter=30)
    print("PSO Best Params:", best_params)
    
    # 3. Particle Filter (SMC)
    print("\n3. Running Sequential Monte Carlo...")
    pf = ParticleFilter(model, obs_model, n_particles=500, ess_threshold=0.5)
    
    # Define Priors based on PSO results + uncertainty
    # For beta, we allow it to vary.
    # For others, we keep them static but uncertain.
    
    def trunc_norm(mean, std, low, high, size=None):
        vals = np.random.normal(mean, std, size)
        return np.clip(vals, low, high)
        
    param_priors = {
        'beta': lambda size: trunc_norm(best_params['beta'], 0.1, 0.0, 2.0, size),
        'sigma': lambda size: trunc_norm(best_params['sigma'], 0.05, 0.05, 1.0, size),
        'gamma': lambda size: trunc_norm(best_params['gamma'], 0.05, 0.05, 1.0, size),
        'rho': lambda size: trunc_norm(best_params['rho'], 0.1, 0.1, 1.0, size),
        'kappa': lambda size: np.random.uniform(2.0, 20.0, size) # High uncertainty on dispersion
    }
    
    # Initial State Priors
    # We'll assume we know N, but initial I is uncertain.
    def init_S_I(size):
        # I ~ Uniform(5, 20)
        I = np.random.uniform(5, 20, size)
        S = N - I
        E = np.zeros(size)
        R = np.zeros(size)
        C = np.zeros(size)
        return np.stack([S, E, I, R, C], axis=1)
        
    # We need to pass columns to PF initialize.
    # Since my PF implementation takes separate priors for compartments...
    # Let's adjust PF or just pass fixed for simplicity, or implement lambda support for full state.
    # The PF `initialize` method takes dict of scalars or callables.
    # Let's just do simple independent priors for now.
    
    pf.initialize(
        initial_state_priors={
            'S': lambda size: N - np.random.uniform(5, 20, size),
            'E': 0.0,
            'I': lambda size: np.random.uniform(5, 20, size),
            'R': 0.0,
            'C': 0.0
        },
        param_priors=param_priors
    )
    
    # Enable Random Walk for Beta
    pf.set_parameter_walk('beta', sigma=0.05) 
    
    # Run SMC
    for t in range(len(observed_data)):
        if t % 10 == 0:
            print(f"  Step {t}/{len(observed_data)}")
        pf.step(t, dt=1.0, observed_data=observed_data[t])
        
    # 4. Diagnostics & Plotting
    print("\n4. Analyzing Results...")
    diag = Diagnostics(pf.get_posterior_estimates())
    
    # Create output dir
    os.makedirs("output", exist_ok=True)
    
    fig = plot_estimates(diag, observed_data, title="SEIR Validation Run")
    fig.savefig("output/validation_plot.png")
    print("Plot saved to output/validation_plot.png")
    
    # Print summary
    final_beta = diag.get_parameter_quantiles('beta')[-1]
    print(f"Final Estimated Beta: {final_beta[1]:.3f} (95% CI: {final_beta[0]:.3f} - {final_beta[2]:.3f})")
    print(f"True Final Beta: {beta_func(t_max)}")

if __name__ == "__main__":
    run_validation()
