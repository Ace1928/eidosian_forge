import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
def check_framework_is_available(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        framework = kwargs.get('framework', 'pt')
        pt_asked_but_not_available = framework == 'pt' and (not is_torch_available())
        tf_asked_but_not_available = framework == 'tf' and (not is_tf_available())
        if (pt_asked_but_not_available or tf_asked_but_not_available) and framework != 'np':
            framework_name = 'PyTorch' if framework == 'pt' else 'TensorFlow'
            raise RuntimeError(f'Requested the {framework_name} framework, but it does not seem installed.')
        return func(*args, **kwargs)
    return wrapper