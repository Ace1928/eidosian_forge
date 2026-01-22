from typing import Callable, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import ListXd, Ragged
@registry.layers('ragged2list.v1')
def ragged2list() -> Model[InT, OutT]:
    """Transform sequences from a ragged format into lists."""
    return Model('ragged2list', forward)