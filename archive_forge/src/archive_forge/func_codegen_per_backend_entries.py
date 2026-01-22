import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def codegen_per_backend_entries() -> str:
    r = []
    for fk in FUNCTIONALITY_KEYS:
        for bc in BACKEND_COMPONENTS:
            r.append(f'    {fk}{bc} = auto()')
    return '\n'.join(r)