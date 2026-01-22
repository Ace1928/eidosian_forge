import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import (
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile
from IPython.core.magic import (
def doctest_hist_op():
    """Test %hist -op

    In [1]: class b(float):
       ...:     pass
       ...:

    In [2]: class s(object):
       ...:     def __str__(self):
       ...:         return 's'
       ...:

    In [3]:

    In [4]: class r(b):
       ...:     def __repr__(self):
       ...:         return 'r'
       ...:

    In [5]: class sr(s,r): pass
       ...:

    In [6]:

    In [7]: bb=b()

    In [8]: ss=s()

    In [9]: rr=r()

    In [10]: ssrr=sr()

    In [11]: 4.5
    Out[11]: 4.5

    In [12]: str(ss)
    Out[12]: 's'

    In [13]:

    In [14]: %hist -op
    >>> class b:
    ...     pass
    ...
    >>> class s(b):
    ...     def __str__(self):
    ...         return 's'
    ...
    >>>
    >>> class r(b):
    ...     def __repr__(self):
    ...         return 'r'
    ...
    >>> class sr(s,r): pass
    >>>
    >>> bb=b()
    >>> ss=s()
    >>> rr=r()
    >>> ssrr=sr()
    >>> 4.5
    4.5
    >>> str(ss)
    's'
    >>>
    """