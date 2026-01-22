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
@my_registry.cats.register('catsie.v2')
def catsie_v2(evil: StrictBool, cute: bool=True, cute_level: int=1) -> str:
    if evil:
        return 'scratch!'
    else:
        if cute_level > 2:
            return 'meow <3'
        return 'meow'