import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@property
def flat_non_out(self) -> Sequence[Argument]:
    ret: List[Argument] = []
    ret.extend(self.flat_positional)
    ret.extend(self.flat_kwarg_only)
    return ret