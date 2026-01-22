from typing import Callable, List, Tuple
from thinc.api import Model, chain, with_array
from thinc.types import Floats1d, Floats2d
from ...tokens import Doc
from ...util import registry
def backprop(dY: Floats2d) -> List[Floats2d]:
    return model.ops.unflatten(dY, lens)