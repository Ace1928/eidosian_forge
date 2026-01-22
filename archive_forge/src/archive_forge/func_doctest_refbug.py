import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
def doctest_refbug():
    """Very nasty problem with references held by multiple runs of a script.
    See: https://github.com/ipython/ipython/issues/141

    In [1]: _ip.clear_main_mod_cache()
    # random

    In [2]: %run refbug

    In [3]: call_f()
    lowercased: hello

    In [4]: %run refbug

    In [5]: call_f()
    lowercased: hello
    lowercased: hello
    """