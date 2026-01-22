import functools
import logging
import os
import sys
import threading
from logging import (
from logging import captureWarnings as _captureWarnings
from typing import Optional
import huggingface_hub.utils as hf_hub_utils
from tqdm import auto as tqdm_lib
def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var TRANSFORMERS_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv('TRANSFORMERS_NO_ADVISORY_WARNINGS', False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)