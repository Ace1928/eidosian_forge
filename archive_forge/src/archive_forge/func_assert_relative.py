from functools import partial
from glob import glob
from distutils.util import convert_path
import distutils.command.build_py as orig
import os
import fnmatch
import textwrap
import io
import distutils.errors
import itertools
import stat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from ..extern.more_itertools import unique_everseen
from ..warnings import SetuptoolsDeprecationWarning
def assert_relative(path):
    if not os.path.isabs(path):
        return path
    from distutils.errors import DistutilsSetupError
    msg = textwrap.dedent('\n        Error: setup script specifies an absolute path:\n\n            %s\n\n        setup() arguments must *always* be /-separated paths relative to the\n        setup.py directory, *never* absolute paths.\n        ').lstrip() % path
    raise DistutilsSetupError(msg)