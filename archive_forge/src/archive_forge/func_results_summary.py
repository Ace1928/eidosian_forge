import copy
import os
import traceback
import warnings
from keras_tuner.src import backend
from keras_tuner.src import config as config_module
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.distribute import utils as dist_utils
from keras_tuner.src.engine import hypermodel as hm_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner_utils
def results_summary(self, num_trials=10):
    """Display tuning results summary.

        The method prints a summary of the search results including the
        hyperparameter values and evaluation results for each trial.

        Args:
            num_trials: Optional number of trials to display. Defaults to 10.
        """
    print('Results summary')
    print(f'Results in {self.project_dir}')
    print('Showing %d best trials' % num_trials)
    print(f'{self.oracle.objective}')
    best_trials = self.oracle.get_best_trials(num_trials)
    for trial in best_trials:
        print()
        trial.summary()