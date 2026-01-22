from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
@registry.initializers('zero_init.v1')
def configure_zero_init() -> Callable[[FloatsXd], FloatsXd]:
    return partial(zero_init)