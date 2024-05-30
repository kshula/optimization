#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>

using namespace std;
using namespace Eigen;
using namespace ceres;

// Define the cost function for the optimizer
struct PortfolioCostFunction : public CostFunction {
    PortfolioCostFunction(const VectorXd& returns, const MatrixXd& cov_matrix, double risk_tolerance)
        : expected_returns(returns), covariance_matrix(cov_matrix), risk_tolerance(risk_tolerance) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(returns.size());
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        Map<const VectorXd> weights(parameters[0], expected_returns.size());
        double portfolio_return = expected_returns.dot(weights);
        double portfolio_risk = weights.transpose() * covariance_matrix * weights;
        residuals[0] = -(portfolio_return - risk_tolerance * portfolio_risk);

        if (jacobians && jacobians[0]) {
            Map<MatrixXd> jacobian(jacobians[0], 1, expected_returns.size());
            jacobian = -expected_returns.transpose() + 2 * risk_tolerance * (weights.transpose() * covariance_matrix);
        }
        return true;
    }

    private:
    const VectorXd& expected_returns;
    const MatrixXd& covariance_matrix;
    double risk_tolerance;
};

int main() {
    // Sample data: expected returns, covariance matrix, and risk tolerance
    VectorXd expected_returns(3);
    expected_returns << 0.1, 0.2, 0.15;

    MatrixXd covariance_matrix(3, 3);
    covariance_matrix << 0.005, -0.010, 0.004,
                         -0.010, 0.040, -0.002,
                          0.004, -0.002, 0.023;

    double risk_tolerance = 0.1;

    // Initial guess for weights
    VectorXd weights = VectorXd::Ones(3) / 3.0;

    // Set up the optimization problem
    Problem problem;
    CostFunction* cost_function = new PortfolioCostFunction(expected_returns, covariance_matrix, risk_tolerance);
    problem.AddResidualBlock(cost_function, nullptr, weights.data());

    // Constraint: sum of weights == 1
    problem.AddResidualBlock(new AutoDiffCostFunction<ceres::SumConstraint, 1, 3>(
        new ceres::SumConstraint(1.0)), nullptr, weights.data());

    // Constraint: weights >= 0
    for (int i = 0; i < weights.size(); ++i) {
        problem.SetParameterLowerBound(weights.data(), i, 0.0);
    }

    // Solve the problem
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Output results
    cout << "Optimal weights: " << weights.transpose() << endl;

    return 0;
}
