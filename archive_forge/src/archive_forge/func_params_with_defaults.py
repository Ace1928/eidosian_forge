from functools import partial
import numpy as np
from . import _catboost
@staticmethod
def params_with_defaults():
    """
        For each valid metric parameter, returns its default value and if this parameter is mandatory.
        Implemented in child classes.

        Returns
        ----------
        valid_params: dict: param_name -> {'default_value': default value or None, 'is_mandatory': bool}
        """
    raise NotImplementedError('Should be overridden by the child class.')