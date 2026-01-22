import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
def inspect_parameter_names(self) -> List[str]:
    unimplemented(f'inspect_parameter_names: {self}')