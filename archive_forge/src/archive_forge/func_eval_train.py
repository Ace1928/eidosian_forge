import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def eval_train(self, feval: Optional[Union[_LGBM_CustomEvalFunction, List[_LGBM_CustomEvalFunction]]]=None) -> List[_LGBM_BoosterEvalMethodResultType]:
    """Evaluate for training data.

        Parameters
        ----------
        feval : callable, list of callable, or None, optional (default=None)
            Customized evaluation function.
            Each evaluation function should accept two parameters: preds, eval_data,
            and return (eval_name, eval_result, is_higher_better) or list of such tuples.

                preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                    The predicted values.
                    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                    If custom objective function is used, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                eval_data : Dataset
                    The training dataset.
                eval_name : str
                    The name of evaluation function (without whitespace).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

        Returns
        -------
        result : list
            List with (train_dataset_name, eval_name, eval_result, is_higher_better) tuples.
        """
    return self.__inner_eval(self._train_data_name, 0, feval)