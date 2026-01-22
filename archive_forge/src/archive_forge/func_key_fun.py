from io import StringIO
import numbers
from typing import (
from Bio import BiopythonDeprecationWarning, StreamModeError
from Bio.Seq import Seq, MutableSeq, UndefinedSequenceError
import warnings
def key_fun(f):
    """Sort on start position."""
    try:
        return int(f.location.start)
    except TypeError:
        return None