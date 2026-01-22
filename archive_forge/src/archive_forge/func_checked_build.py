import abc
from dataclasses import dataclass, field
import functools
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import ExperimentalAPI
@functools.wraps(fn)
def checked_build(self, framework, **kwargs):
    if framework not in accepted:
        raise ValueError(f'This config does not support framework {framework}. Only frameworks in {accepted} are supported.')
    return fn(self, framework, **kwargs)