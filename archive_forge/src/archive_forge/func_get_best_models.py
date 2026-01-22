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
def get_best_models(self, num_models=1):
    """Returns the best model(s), as determined by the objective.

        This method is for querying the models trained during the search.
        For best performance, it is recommended to retrain your Model on the
        full dataset using the best hyperparameters found during `search`,
        which can be obtained using `tuner.get_best_hyperparameters()`.

        Args:
            num_models: Optional number of best models to return.
                Defaults to 1.

        Returns:
            List of trained models sorted from the best to the worst.
        """
    best_trials = self.oracle.get_best_trials(num_models)
    models = [self.load_model(trial) for trial in best_trials]
    return models