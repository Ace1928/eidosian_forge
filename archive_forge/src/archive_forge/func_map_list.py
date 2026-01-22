from typing import Callable, List, Optional, Tuple, TypeVar
from ..model import Model
def map_list(layer: Model[InT, OutT]) -> Model[List[InT], List[OutT]]:
    """Create a model that maps a child layer across list inputs."""
    return Model('map_list', forward, layers=[layer], init=init)