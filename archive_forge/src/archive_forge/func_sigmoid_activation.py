from typing import Callable, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import FloatsXdT
@registry.layers('sigmoid_activation.v1')
def sigmoid_activation() -> Model[FloatsXdT, FloatsXdT]:
    return Model('sigmoid_activation', forward)