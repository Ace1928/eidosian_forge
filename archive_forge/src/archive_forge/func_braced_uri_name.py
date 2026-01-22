from typing import Any, Optional
from ..helpers import QNAME_PATTERN
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
@property
def braced_uri_name(self) -> str:
    return 'Q{%s}%s' % (self.uri, self.local_name)