import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType,
from .callback import _EvalResultDict, record_evaluation
from .compat import (SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
from .engine import train
@property
def objective_(self) -> Union[str, _LGBM_ScikitCustomObjectiveFunction]:
    """:obj:`str` or :obj:`callable`: The concrete objective used while fitting this model."""
    if not self.__sklearn_is_fitted__():
        raise LGBMNotFittedError('No objective found. Need to call fit beforehand.')
    return self._objective