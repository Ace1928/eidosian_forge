import inspect
import pickle
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import catalogue
import numpy
import pytest
import thinc.config
from thinc.api import Config, Model, NumpyOps, RAdam
from thinc.config import ConfigValidationError
from thinc.types import Generator, Ragged
from thinc.util import partial
from .util import make_tempdir
@thinc.registry.optimizers.register('my_cool_optimizer.v1')
def make_my_optimizer(learn_rate: List[float], beta1: float):
    return RAdam(learn_rate, beta1=beta1)