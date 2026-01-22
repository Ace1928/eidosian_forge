from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
def eligible_for_binary_scalar_specialization(g: NativeFunctionsGroup) -> bool:
    num_tensors = sum((1 for a in g.functional.func.arguments.flat_non_out if a.type.is_tensor_like()))
    return num_tensors == 2