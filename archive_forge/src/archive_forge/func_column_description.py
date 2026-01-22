from __future__ import print_function
import os
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from .utils import make_dirs_if_not_exists
def column_description(self):
    """
        Args:
            :return: (str) Path to the column description.

        """
    return self._column_description