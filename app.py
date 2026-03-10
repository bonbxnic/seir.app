import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import tempfile
import matplotlib.pyplot as plt

# Import framework components
from seir_framework.model.seir import SEIRModel
from seir_framework.inference.likelihood import NegativeBinomialLikelihood
from seir_framework.inference.smc import ParticleFilter
from seir_framework.inference.pso import AdaptivePSO
from seir_framework.utils.diagnostics import Diagnostics
from seir_framework.utils.viz import plot_estimates, animate_results

# Page Config
st.set_page_config(page_title="SEIR Analysis Tool", layout="wide")

st.title("🦠 Epidemic Analysis & Estimation Tool")
st.markdown("""
Upload your daily case data (CSV) to estimate transmission rates ($R_t$) and visualize the outbreak dynamics.
""")

# Sidebar
st.sidebar.header("Configuration")
population_size = st.sidebar.number_input("Population Size", min_value=1000, value=100000, step=1000)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Assumptions")
incubation_period = st.sidebar.slider("Incubation Period (days)", 2.0, 14.0, 5.0)
infectious_period = st.sidebar.slider("Infectious Period (days)", 2.0, 21.0, 10.0)

# Main Area
uploaded_file = st.file_uploader("Upload CSV File (must contain a 'cases' column)", type=['csv'])

if uploaded_file is not None:
    # Load and display data
    try:
        df = pd.read_csv(uploaded_file)
        if 'cases' not in df.columns:
            st.error("CSV must contain a 'cases' column!")
        else:
            st.success("File uploaded successfully!")
            
            # Show preview
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(df.head())
            with col2:
                st.line_chart(df['cases'])
            
            # Run Button
            if st.button("🚀 Run Estimation & Simulation"):
                
                # Setup progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Prepare data
                observed_data = df['cases'].values
                observed_data = np.nan_to_num(observed_data)
                days = len(observed_data)
                
                # --- Step 1: Initialization ---
                status_text.text("Initializing Model...")
                time.sleep(0.5)
                
                sigma = 1.0 / incubation_period
                gamma = 1.0 / infectious_period
                default_params = {'beta': 0.5, 'sigma': sigma, 'gamma': gamma}
                
                model = SEIRModel(N=population_size, params=default_params)
                obs_model = NegativeBinomialLikelihood()
                
                # --- Step 2: PSO ---
                status_text.text("Estimating initial parameters (Running PSO)...")
                progress_bar.progress(10)
                
                param_bounds = {
                    'beta': (0.1, 3.0),
                    'sigma': (sigma * 0.8, sigma * 1.2), # Constrain near user input
                    'gamma': (gamma * 0.8, gamma * 1.2),
                    'rho': (0.1, 1.0)
                }
                
                i0 = max(1, observed_data[0])
                initial_state = np.array([population_size - 2*i0, i0, i0, 0, 0])
                
                pso = AdaptivePSO(model, observed_data, obs_model, param_bounds, population_size=40)
                best_params = pso.optimize(initial_state, max_iter=30)
                
                st.write("Initial Estimates:", best_params)
                progress_bar.progress(30)
                
                # --- Step 3: SMC ---
                status_text.text("Running Particle Filter (SMC)...")
                
                pf = ParticleFilter(model, obs_model, n_particles=500, ess_threshold=0.5)
                
                def trunc_norm(mean, std, low, high, size=None):
                    vals = np.random.normal(mean, std, size)
                    return np.clip(vals, low, high)

                param_priors = {
                    'beta': lambda size: trunc_norm(best_params['beta'], 0.2, 0.0, 3.0, size),
                    'sigma': lambda size: trunc_norm(best_params['sigma'], 0.05, 0.05, 1.0, size),
                    'gamma': lambda size: trunc_norm(best_params['gamma'], 0.05, 0.05, 1.0, size),
                    'rho': lambda size: trunc_norm(best_params['rho'], 0.1, 0.1, 1.0, size),
                    'kappa': lambda size: np.random.uniform(2.0, 20.0, size)
                }

                pf.initialize(
                    initial_state_priors={'S': initial_state[0], 'E': initial_state[1], 'I': initial_state[2], 'R': 0, 'C': 0},
                    param_priors=param_priors
                )
                pf.set_parameter_walk('beta', sigma=0.05)
                
                # Run Loop
                for t in range(days):
                    obs = observed_data[t]
                    pf.step(t, dt=1.0, observed_data=obs)
                    
                    # Update progress
                    pct = 30 + int(70 * (t / days))
                    progress_bar.progress(pct)
                    status_text.text(f"Simulating Day {t+1}/{days}...")
                
                progress_bar.progress(100)
                status_text.text("Analysis Complete!")
                
                # --- Results ---
                diag = Diagnostics(pf.get_posterior_estimates())
                
                st.markdown("### 📊 Results")
                
                # Static Plot
                fig = plot_estimates(diag, observed_data)
                st.pyplot(fig)
                
                # Animation
                st.markdown("### 🎬 Simulation Animation")
                with st.spinner("Generating animation..."):
                    # Use temp file for gif
                    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
                        animate_results(diag, observed_data, tmp.name)
                        st.image(tmp.name)
                        
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
    
    # Download sample button
    if st.button("Create Sample Data"):
        from generate_synthetic_data import generate_data
        path = generate_data('data/sample_data.csv')
        st.success(f"Sample data created at {path}. You can download it if running locally, or just find it in the folder.")
