package optimize;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.optim.*;
import org.apache.commons.math3.optim.linear.*;
import java.util.Arrays;

public class PortfolioOptimization {
    public static void main(String[] args) {
        // Sample data: expected returns, covariance matrix, and risk tolerance
        double[] expectedReturnsArray = { 0.1, 0.2, 0.15 };
        double[][] covarianceMatrixArray = {
            { 0.005, -0.010, 0.004 },
            { -0.010, 0.040, -0.002 },
            { 0.004, -0.002, 0.023 }
        };
        double riskTolerance = 0.1;

        // Convert arrays to Apache Commons Math matrices and vectors
        RealVector expectedReturns = MatrixUtils.createRealVector(expectedReturnsArray);
        RealMatrix covarianceMatrix = MatrixUtils.createRealMatrix(covarianceMatrixArray);

        // Number of assets
        int n = expectedReturns.getDimension();

        // Initial guess for weights
        double[] initialGuess = new double[n];
        Arrays.fill(initialGuess, 1.0 / n);

        // Objective function: maximize return - riskTolerance * risk
        MultivariateFunction objectiveFunction = new MultivariateFunction() {
            @Override
            public double value(double[] weights) {
                RealVector weightsVector = MatrixUtils.createRealVector(weights);
                double portfolioReturn = expectedReturns.dotProduct(weightsVector);
                double portfolioRisk = weightsVector.dotProduct(covarianceMatrix.operate(weightsVector));
                return -(portfolioReturn - riskTolerance * portfolioRisk);
            }
        };

        // Constraints: sum(weights) = 1 and weights >= 0
        LinearConstraint[] constraints = new LinearConstraint[n + 1];
        // Sum of weights = 1
        double[] sumEqualsOne = new double[n];
        Arrays.fill(sumEqualsOne, 1.0);
        constraints[0] = new LinearConstraint(sumEqualsOne, Relationship.EQ, 1.0);

        // Weights >= 0
        for (int i = 0; i < n; i++) {
            double[] nonNegativeConstraint = new double[n];
            nonNegativeConstraint[i] = 1.0;
            constraints[i + 1] = new LinearConstraint(nonNegativeConstraint, Relationship.GEQ, 0.0);
        }

        // Optimize
        SimplexOptimizer optimizer = new SimplexOptimizer(1e-10, 1e-30);
        PointValuePair solution = optimizer.optimize(
            new MaxIter(1000),
            GoalType.MINIMIZE,
            new InitialGuess(initialGuess),
            new ObjectiveFunction(objectiveFunction),
            new LinearConstraintSet(constraints)
        );

        // Display the results
        double[] optimalWeights = solution.getPoint();
        System.out.println("Optimal weights: " + Arrays.toString(optimalWeights));

        RealVector optimalWeightsVector = MatrixUtils.createRealVector(optimalWeights);
        double optimalReturn = expectedReturns.dotProduct(optimalWeightsVector);
        double optimalRisk = optimalWeightsVector.dotProduct(covarianceMatrix.operate(optimalWeightsVector));

        System.out.println("Expected return: " + optimalReturn);
        System.out.println("Portfolio risk: " + optimalRisk);
    }
}
