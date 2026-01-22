from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import XY_YZ_OutT
from ..util import get_width
@registry.layers('chain.v1')
def chain_no_types(*layer: Model) -> Model:
    return chain(*layer)