from typing import Any, Optional
from ..helpers import QNAME_PATTERN
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
@property
def expanded_name(self) -> str:
    return '{%s}%s' % (self.uri, self.local_name) if self.uri else self.local_name