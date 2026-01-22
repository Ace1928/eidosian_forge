import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def compress_dhist(dh):
    """Compress a directory history into a new one with at most 20 entries.

    Return a new list made from the first and last 10 elements of dhist after
    removal of duplicates.
    """
    head, tail = (dh[:-10], dh[-10:])
    newhead = []
    done = set()
    for h in head:
        if h in done:
            continue
        newhead.append(h)
        done.add(h)
    return newhead + tail