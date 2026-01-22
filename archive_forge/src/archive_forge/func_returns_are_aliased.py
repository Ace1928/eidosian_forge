import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def returns_are_aliased(self) -> bool:
    return any((r for r in self.returns if r.annotation is not None and r.annotation.is_write))