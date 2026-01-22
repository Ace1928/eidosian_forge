import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
def parse_returns(return_decl: str) -> Tuple[Return, ...]:
    """
    Input: '()'
    Output: []
    """
    if return_decl == '()':
        return ()
    if return_decl[0] == '(' and return_decl[-1] == ')':
        return_decl = return_decl[1:-1]
    return tuple((Return.parse(arg) for arg in return_decl.split(', ')))