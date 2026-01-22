import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
def get_kernels(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> Dict[ETKernelKey, BackendMetadata]:
    if isinstance(g, NativeFunction):
        f = g
    elif isinstance(g, NativeFunctionsGroup):
        f = g.functional
    else:
        assert_never(g)
    if f.func.name not in self.index:
        return {}
    return self.index[f.func.name]