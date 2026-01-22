from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def get_owning_type(t: CType) -> Tuple[CType, Callable[[str], str]]:
    if t == BaseCType(tensorListT):
        return (VectorCType(BaseCType(tensorT)), lambda x: f'{x}.vec()')
    if t == BaseCType(iTensorListRefT):
        return (VectorCType(BaseCType(tensorT)), lambda x: f'{{{x}.begin(), {x}.end()}}')
    return (t, lambda x: x)