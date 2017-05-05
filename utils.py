from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter

def hpolib_to_skopt_bounds(problem):
    bounds = problem.get_meta_information()["bounds"]
    hyperparams = problem.configuration_space.get_hyperparameters()

    new_bounds = []
    for i, bound in enumerate(bounds):
        if len(bound) == 2:
            l_b, u_b = bound
            if isinstance(hyperparams[i], FloatHyperparameter):
                new_bounds.append([float(l_b), float(u_b)])
            elif isinstance(hyperparams[i], IntegerHyperparameter):
                new_bounds.append([int(l_b), int(u_b)])
    return new_bounds
