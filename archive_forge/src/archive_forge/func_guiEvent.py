from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
@property
def guiEvent(self):
    if self._guiEvent_deleted:
        _api.warn_deprecated('3.8', message='Accessing guiEvent outside of the original GUI event handler is unsafe and deprecated since %(since)s; in the future, the attribute will be set to None after quitting the event handler.  You may separately record the value of the guiEvent attribute at your own risk.')
    return self._guiEvent