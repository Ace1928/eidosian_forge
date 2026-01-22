import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def set_inputs(self, inputs):
    self.inputs = {**self.model.prepare_inputs_for_generation(**inputs), **inputs}
    self.inputs['input_features'] = self.inputs.pop('inputs')