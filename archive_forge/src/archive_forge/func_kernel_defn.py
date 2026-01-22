from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
def kernel_defn(self) -> str:
    return f'void {self.kernel_name}(TensorIteratorBase& iter, {', '.join((a.defn() for a in self.arguments()))})'