import numpy as np
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
def populate_space(self, trial_id):
    """Fill the hyperparameter space with values.

        Args:
            trial_id: A string, the ID for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            should be one of "RUNNING" (the trial can start normally), "IDLE"
            (the oracle is waiting on something and cannot create a trial), or
            "STOPPED" (the oracle has finished searching and no new trial should
            be created).
        """
    completed_trials = [t for t in self.trials.values() if t.status == 'COMPLETED']
    dimensions = len(self.hyperparameters.space)
    num_initial_points = self.num_initial_points or max(3 * dimensions, 3)
    if len(completed_trials) < num_initial_points:
        return self._random_populate_space()
    x, y = self._vectorize_trials()
    x = np.nan_to_num(x, posinf=0, neginf=0)
    y = np.nan_to_num(y, posinf=0, neginf=0)
    self.gpr.fit(x, y)

    def _upper_confidence_bound(x):
        x = x.reshape(1, -1)
        mu, sigma = self.gpr.predict(x, return_std=True)
        return mu - self.beta * sigma
    optimal_val = float('inf')
    optimal_x = None
    num_restarts = 50
    bounds = self._get_hp_bounds()
    x_seeds = self._random_state.uniform(bounds[:, 0], bounds[:, 1], size=(num_restarts, bounds.shape[0]))
    for x_try in x_seeds:
        result = scipy.optimize.minimize(_upper_confidence_bound, x0=x_try, bounds=bounds, method='L-BFGS-B')
        result_fun = result.fun if np.isscalar(result.fun) else result.fun[0]
        if result_fun < optimal_val:
            optimal_val = result_fun
            optimal_x = result.x
    values = self._vector_to_values(optimal_x)
    return {'status': trial_module.TrialStatus.RUNNING, 'values': values}