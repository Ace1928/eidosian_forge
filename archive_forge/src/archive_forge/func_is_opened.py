from __future__ import print_function
import os
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from .utils import make_dirs_if_not_exists
def is_opened(self):
    return self._file is not None