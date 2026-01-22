import importlib
import math
import os
import sys
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
@staticmethod
def format_num(n: Union[int, float, str]) -> str:
    """Add additional padding to the formatted numbers."""
    should_be_padded = isinstance(n, (float, str))
    if not isinstance(n, str):
        n = _tqdm.format_num(n)
        assert isinstance(n, str)
    if should_be_padded and 'e' not in n:
        if '.' not in n and len(n) < _PAD_SIZE:
            try:
                _ = float(n)
            except ValueError:
                return n
            n += '.'
        n += '0' * (_PAD_SIZE - len(n))
    return n