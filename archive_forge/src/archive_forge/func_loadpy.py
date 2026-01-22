import inspect
import io
import os
import re
import sys
import ast
from itertools import chain
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from IPython.core.error import TryNext, StdinNotImplementedError, UsageError
from IPython.core.macro import Macro
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.oinspect import find_file, find_source_lines
from IPython.core.release import version
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import get_py_filename
from warnings import warn
from logging import error
from IPython.utils.text import get_text_list
@line_magic
def loadpy(self, arg_s):
    """Alias of `%load`

        `%loadpy` has gained some flexibility and dropped the requirement of a `.py`
        extension. So it has been renamed simply into %load. You can look at
        `%load`'s docstring for more info.
        """
    self.load(arg_s)