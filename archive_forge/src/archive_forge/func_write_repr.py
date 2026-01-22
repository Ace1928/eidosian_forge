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
def write_repr(self, out, visited):
    proxy = self.proxyval(visited)
    out.write(proxy)