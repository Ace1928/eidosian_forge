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
@my_registry.cats.register('catsie.v1')
def catsie_v1(evil: StrictBool, cute: bool=True) -> str:
    if evil:
        return 'scratch!'
    else:
        return 'meow'