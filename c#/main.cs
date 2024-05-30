using System;
using MathNet.Numerics.LinearAlgebra;
using Accord.Math.Optimization;

class PortfolioOptimization
{
    static void Main()
    {
        // Sample data: expected returns, covariance matrix, and risk tolerance
        double[] expectedReturnsArray = { 0.1, 0.2, 0.15 };
        double[,] covarianceMatrixArray = {
            { 0.005, -0.010, 0.004 },
            { -0.010, 0.040, -0.002 },
            { 0.004, -0.002, 0.023 }
        };
        double riskTolerance = 0.1;

        // Convert arrays to MathNet matrices and vectors
        var expectedReturns = Vector<double>.Build.DenseOfArray(expectedReturnsArray);
        var covarianceMatrix = Matrix<double>.Build.DenseOfArray(covarianceMatrixArray);

        // Number of assets
        int n = expectedReturns.Count;

        // Define the objective function
        Func<double[], double> objectiveFunction = (weights) =>
        {
            var weightsVector = Vector<double>.Build.DenseOfArray(weights);
            double portfolioReturn = expectedReturns.DotProduct(weightsVector);
            double portfolioRisk = weightsVector * covarianceMatrix * weightsVector;
            return -(portfolioReturn - riskTolerance * portfolioRisk);
        };

        // Initial guess for weights
        double[] initialGuess = new double[n];
        for (int i = 0; i < n; i++)
            initialGuess[i] = 1.0 / n;

        // Constraints: sum(weights) = 1 and weights >= 0
        var constraints = new List<NonlinearConstraint>
        {
            new NonlinearConstraint(n, 
                function: (weights) => weights.Sum(), 
                shouldBe: 1.0),
            new NonlinearConstraint(n, 
                function: (weights) => weights.Min(), 
                shouldBeGreaterThanOrEqualTo: 0.0)
        };

        // Define the optimizer
        var optimizer = new Cobyla(n, constraints.ToArray());
        optimizer.Function = objectiveFunction;

        // Solve the optimization problem
        bool success = optimizer.Minimize(initialGuess);

        if (success)
        {
            double[] optimalWeights = optimizer.Solution;
            Console.WriteLine("Optimal weights: " + string.Join(", ", optimalWeights));

            var optimalWeightsVector = Vector<double>.Build.DenseOfArray(optimalWeights);
            double optimalReturn = expectedReturns.DotProduct(optimalWeightsVector);
            double optimalRisk = optimalWeightsVector * covarianceMatrix * optimalWeightsVector;

            Console.WriteLine("Expected return: " + optimalReturn);
            Console.WriteLine("Portfolio risk: " + optimalRisk);
        }
        else
        {
            Console.WriteLine("Optimization failed.");
        }
    }
}
