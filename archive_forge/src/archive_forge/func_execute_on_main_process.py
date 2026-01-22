import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
@wraps(function)
def execute_on_main_process(self, *args, **kwargs):
    if getattr(self, 'main_process_only', False):
        return PartialState().on_main_process(function)(self, *args, **kwargs)
    else:
        return function(self, *args, **kwargs)