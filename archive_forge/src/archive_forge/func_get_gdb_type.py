from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
@classmethod
def get_gdb_type(cls):
    return gdb.lookup_type(cls._typename).pointer()