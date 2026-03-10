# SEIR Epidemic Analysis Tool

This is a powerful tool for simulating and estimating epidemic dynamics using the SEIR model.

## How to Run

**Method 1: The Easy Way (App Launcher)**
Double-click the `start_app.bat` file in this folder. This will automatically launch the tool in your web browser.

**Method 2: The Manual Way**
Open a terminal in this folder and run:
```bash
streamlit run app.py
```

## Features
- **Upload Your Data**: Drag and drop CSV files with daily case counts.
- **Real-time Estimation**: Uses Particle Swarm Optimization (PSO) and Particle Filters (SMC) to fit the model.
- **Interactive Simulation**: Visualize the spread of the disease and the effective transmission rate ($R_t$) over time.
- **Animated Results**: Generate and download GIFs of the outbreak dynamics.

## Data Format
Your CSV file must have a column named `cases` representing the number of new cases per day.
Example:
```csv
date,cases
2024-01-01,5
2024-01-02,8
...
```
You can generate sample data directly within the app if you don't have any.
