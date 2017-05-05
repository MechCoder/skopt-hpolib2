"""
Machine learning benchmarks.
"""
import argparse
import numpy as np

from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetworkOnCIFAR10
from hpolib.benchmarks.ml.fully_connected_network import FCNetOnMnist
from hpolib.benchmarks.ml.logistic_regression import LogisticRegressionOnMnist

from skopt import dump
from skopt import forest_minimize
from skopt import gp_minimize
from utils import hpolib_to_skopt_bounds

def run(problem, optimizer, n_calls, n_runs):
    if problem == "fc":
        problem = FCNetOnMnist()
        bounds = hpolib_to_skopt_bounds(problem)

    elif problem == "cnn":
        problem = ConvolutionalNeuralNetworkOnCIFAR10()
        bounds = hpolib_to_skopt_bounds(problem)

    elif problem == "lr":
        problem = LogisticRegressionOnMnist()
        bounds = hpolib_to_skopt_bounds(problem)
    else:
        raise ValueError("Problem should be either fc | cnn | lr")

    func = lambda x: problem.objective_function(configuration=x)["function_value"]

    if optimizer == "gp":
        opt = gp_minimize
    elif optimizer == "forest":
        opt = forest_minimize

    min_vals = []
    for n_run in range(n_runs):
        res = opt(
            func, bounds, n_calls=n_calls, n_random_starts=1,
            verbose=1, random_state=n_run)
        del res["specs"]
        dump(res, "%s_%d.pkl" % (optimizer, n_run))
        min_vals.append(res.fun)

    print(min_vals)
    return np.min(min_vals), np.std(min_vals)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problem', nargs="?", default="lr", type=str, help="lr | cnn | fc")
    parser.add_argument(
        '--optimizer', nargs="?", default="gp", type=str, help="gp | forest")
    parser.add_argument(
        '--n_calls', nargs="?", default="50", type=int, help="Number of calls")
    parser.add_argument(
        '--n_runs', nargs="?", default="5", type=int, help="Number of runs")
    args = parser.parse_args()
    min_vals, std_vals = run(args.problem, args.optimizer, args.n_calls, args.n_runs)
