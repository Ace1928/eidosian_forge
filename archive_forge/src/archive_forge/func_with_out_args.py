import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def with_out_args(self, outs: List[Argument]) -> 'Arguments':
    assert len(self.out) == 0
    return dataclasses.replace(self, out=tuple(outs))