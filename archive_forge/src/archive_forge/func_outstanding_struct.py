import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def outstanding_struct():
    if not lines_a and (not lines_b):
        return
    elif ch_a and (not ch_b):
        yield (lines_a,)
    elif ch_b and (not ch_a):
        yield (lines_b,)
    elif lines_a == lines_b:
        yield (lines_a,)
    else:
        yield (lines_a, lines_b)