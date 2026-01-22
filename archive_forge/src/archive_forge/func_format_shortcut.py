import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
@staticmethod
def format_shortcut(key_sequence):
    """
        Convert a shortcut string from the notation used in rc config to the
        standard notation for displaying shortcuts, e.g. 'ctrl+a' -> 'Ctrl+A'.
        """
    return key_sequence if len(key_sequence) == 1 else re.sub('\\+[A-Z]', '+Shift\\g<0>', key_sequence).title()