from decimal import Decimal
from urllib.parse import urlparse
from typing import Union
from ..helpers import collapse_white_spaces, WRONG_ESCAPE_PATTERN
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
from .numeric import Integer

    Class for xs:anyURI data.

    :param value: a string or an untyped atomic instance.
    