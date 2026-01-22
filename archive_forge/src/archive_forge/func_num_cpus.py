import os
import platform
import pprint
import sys
import subprocess
from pathlib import Path
from IPython.core import release
from IPython.utils import _sysinfo, encoding
def num_cpus():
    """DEPRECATED

    Return the effective number of CPUs in the system as an integer.

    This cross-platform function makes an attempt at finding the total number of
    available CPUs in the system, as returned by various underlying system and
    python calls.

    If it can't find a sensible answer, it returns 1 (though an error *may* make
    it return a large positive number that's actually incorrect).
    """
    import warnings
    warnings.warn('`num_cpus` is deprecated since IPython 8.0. Use `os.cpu_count` instead.', DeprecationWarning, stacklevel=2)
    return os.cpu_count() or 1