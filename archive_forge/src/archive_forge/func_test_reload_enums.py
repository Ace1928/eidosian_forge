import os
import platform
import pytest
import sys
import tempfile
import textwrap
import shutil
import random
import time
import traceback
from io import StringIO
from dataclasses import dataclass
import IPython.testing.tools as tt
from unittest import TestCase
from IPython.extensions.autoreload import AutoreloadMagics
from IPython.core.events import EventManager, pre_run_cell
from IPython.testing.decorators import skipif_not_numpy
from IPython.core.interactiveshell import ExecutionInfo
def test_reload_enums(self):
    mod_name, mod_fn = self.new_module(textwrap.dedent("\n                                from enum import Enum\n                                class MyEnum(Enum):\n                                    A = 'A'\n                                    B = 'B'\n                            "))
    self.shell.magic_autoreload('2')
    self.shell.magic_aimport(mod_name)
    self.write_file(mod_fn, textwrap.dedent("\n                                from enum import Enum\n                                class MyEnum(Enum):\n                                    A = 'A'\n                                    B = 'B'\n                                    C = 'C'\n                            "))
    with tt.AssertNotPrints('[autoreload of %s failed:' % mod_name, channel='stderr'):
        self.shell.run_code('pass')