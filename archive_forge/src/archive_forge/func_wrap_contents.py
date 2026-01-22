import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def wrap_contents(self):
    """Ensure page is in a balanced graphics state."""
    push, pop = self._count_q_balance()
    if push > 0:
        prepend = b'q\n' * push
        TOOLS._insert_contents(self, prepend, False)
    if pop > 0:
        append = b'\nQ' * pop + b'\n'
        TOOLS._insert_contents(self, append, True)