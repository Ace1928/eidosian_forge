import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
@on_main_process
def log_images(self, values: dict, step: Optional[int]=None, **kwargs):
    """
        Logs `images` to the current run.

        Args:
            values (`Dict[str, List[Union[np.ndarray, PIL.Image]]`):
                Values to be logged as key-value pairs. The values need to have type `List` of `np.ndarray` or
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `clearml.Logger.report_image` method.
        """
    clearml_logger = self.task.get_logger()
    for k, v in values.items():
        title, series = ClearMLTracker._get_title_series(k)
        clearml_logger.report_image(title=title, series=series, iteration=step, image=v, **kwargs)