import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@property
def post_self_positional_mutable(self) -> Sequence[Argument]:
    return [a for a in self.post_self_positional if a.is_write]