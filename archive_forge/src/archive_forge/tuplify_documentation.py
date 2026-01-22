from typing import Any, Optional, Tuple, TypeVar
from ..config import registry
from ..model import Model
Send a separate copy of the input to each child layer, and join the
    outputs of the children into a tuple on the way out.

    Typically used to provide both modified data and the original input to a
    downstream layer.
    