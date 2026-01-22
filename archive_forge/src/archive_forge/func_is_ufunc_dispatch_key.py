import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def is_ufunc_dispatch_key(dk: DispatchKey) -> bool:
    return dk in UFUNC_DISPATCH_KEYS