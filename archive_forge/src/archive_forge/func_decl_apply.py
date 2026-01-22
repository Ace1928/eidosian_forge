from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
def decl_apply(self) -> str:
    args_str = ', '.join((a.decl() for a in self.arguments().apply))
    return f'{self.returns_type().cpp_type()} operator()({args_str}) const'