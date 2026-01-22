from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
@registry.initializers('normal_init.v1')
def configure_normal_init(*, mean: float=0) -> Callable[[FloatsXd], FloatsXd]:
    return partial(normal_init, mean=mean)