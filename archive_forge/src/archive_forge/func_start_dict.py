import collections.abc
import re
from typing import (
import warnings
from io import BytesIO
from datetime import datetime
from base64 import b64encode, b64decode
from numbers import Integral
from types import SimpleNamespace
from functools import singledispatch
from fontTools.misc import etree
from fontTools.misc.textTools import tostr
def start_dict(self: PlistTarget) -> None:
    d = self._dict_type()
    self.add_object(d)
    self.stack.append(d)