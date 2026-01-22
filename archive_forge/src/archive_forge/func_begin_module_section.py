import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def begin_module_section(self, modname):
    self.print(modname)
    self.print('-' * len(modname))
    self.print()