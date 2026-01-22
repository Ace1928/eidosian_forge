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
@skipif_not_numpy
def test_comparing_numpy_structures(self):
    self.shell.magic_autoreload('2')
    self.shell.run_code('1+1')
    mod_name, mod_fn = self.new_module(textwrap.dedent('\n                                import numpy as np\n                                class MyClass:\n                                    a = (np.array((.1, .2)),\n                                         np.array((.2, .3)))\n                            '))
    self.shell.run_code('from %s import MyClass' % mod_name)
    self.shell.run_code('first = MyClass()')
    self.write_file(mod_fn, textwrap.dedent('\n                                import numpy as np\n                                class MyClass:\n                                    a = (np.array((.3, .4)),\n                                         np.array((.5, .6)))\n                            '))
    with tt.AssertNotPrints('[autoreload of %s failed:' % mod_name, channel='stderr'):
        self.shell.run_code('pass')