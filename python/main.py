import numpy as np
import pandas as pd
import cvxpy as cp

# Sample data: expected returns, covariance matrix, and risk tolerance
expected_returns = np.array([0.1, 0.2, 0.15])  # Example expected returns for 3 assets
covariance_matrix = np.array([
    [0.005, -0.010, 0.004],
    [-0.010, 0.040, -0.002],
    [0.004, -0.002, 0.023]
])  # Example covariance matrix for 3 assets

risk_tolerance = 0.1  # Example risk tolerance

# Number of assets
n = len(expected_returns)

# Define the optimization variables
weights = cp.Variable(n)

# Define the expected portfolio return
portfolio_return = expected_returns @ weights

# Define the portfolio risk (variance)
portfolio_risk = cp.quad_form(weights, covariance_matrix)

# Define the objective function to maximize return for a given risk tolerance
objective = cp.Maximize(portfolio_return - risk_tolerance * portfolio_risk)

# Define the constraints
constraints = [
    cp.sum(weights) == 1,  # Weights sum to 1
    weights >= 0  # No short selling (weights must be non-negative)
]

# Formulate the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the optimization problem
problem.solve()

# Display the results
print("Optimal weights:", weights.value)
print("Expected return:", portfolio_return.value)
print("Portfolio risk:", portfolio_risk.value)
