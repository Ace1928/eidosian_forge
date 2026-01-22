import re
import warnings
import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name
def exclude_from_weight_decay(self, var_list=None, var_names=None):
    """Exclude variables from weight decay.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `tf.Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
    if hasattr(self, '_built') and self._built:
        raise ValueError('`exclude_from_weight_decay()` can only be configued before the optimizer is built.')
    if var_list:
        self._exclude_from_weight_decay = [self._var_key(variable) for variable in var_list]
    else:
        self._exclude_from_weight_decay = []
    self._exclude_from_weight_decay_names = var_names or []