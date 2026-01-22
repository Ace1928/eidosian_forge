from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
@registry.initializers('glorot_uniform_init.v1')
def configure_glorot_uniform_init() -> Callable[[Shape], FloatsXd]:
    return partial(glorot_uniform_init)