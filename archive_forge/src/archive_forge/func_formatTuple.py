from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
@property
def formatTuple(self):
    """Return the formatVersion as a tuple of (major, minor).

        .. versionadded:: 5.0
        """
    if self.formatVersion is None:
        return (5, 0)
    numbers = (int(i) for i in self.formatVersion.split('.'))
    major = next(numbers)
    minor = next(numbers, 0)
    return (major, minor)