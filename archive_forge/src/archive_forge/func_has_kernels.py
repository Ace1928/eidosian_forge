import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
def has_kernels(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> bool:
    m = self.get_kernels(g)
    return m is not None