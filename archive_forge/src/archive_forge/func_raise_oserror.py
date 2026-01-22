from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def raise_oserror(error):
    deprecate('raise_oserror', 12, action="It is only useful for translating error codes returned by a codec's decode() method, which ImageFile already does automatically.")
    raise _get_oserror(error, encoder=False)