from __future__ import print_function
import os
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from .utils import make_dirs_if_not_exists
def get_separator(self):
    """
        Args:
            :return: (str) Delimiter for data used when we saved fold to file.

        """
    return self._sep