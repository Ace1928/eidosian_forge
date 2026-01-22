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
def search_space_summary(self, extended=False):
    """Print search space summary.

        The methods prints a summary of the hyperparameters in the search
        space, which can be called before calling the `search` method.

        Args:
            extended: Optional boolean, whether to display an extended summary.
                Defaults to False.
        """
    print('Search space summary')
    hp = self.oracle.get_space()
    print(f'Default search space size: {len(hp.space)}')
    for p in hp.space:
        config = p.get_config()
        name = config.pop('name')
        print(f'{name} ({p.__class__.__name__})')
        print(config)