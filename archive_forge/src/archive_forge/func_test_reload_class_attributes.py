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
def test_reload_class_attributes(self):
    self.shell.magic_autoreload('2')
    mod_name, mod_fn = self.new_module(textwrap.dedent("\n                                class MyClass:\n\n                                    def __init__(self, a=10):\n                                        self.a = a\n                                        self.b = 22 \n                                        # self.toto = 33\n\n                                    def square(self):\n                                        print('compute square')\n                                        return self.a*self.a\n                            "))
    self.shell.run_code('from %s import MyClass' % mod_name)
    self.shell.run_code('first = MyClass(5)')
    self.shell.run_code('first.square()')
    with self.assertRaises(AttributeError):
        self.shell.run_code('first.cube()')
    with self.assertRaises(AttributeError):
        self.shell.run_code('first.power(5)')
    self.shell.run_code('first.b')
    with self.assertRaises(AttributeError):
        self.shell.run_code('first.toto')
    self.write_file(mod_fn, textwrap.dedent("\n                            class MyClass:\n\n                                def __init__(self, a=10):\n                                    self.a = a\n                                    self.b = 11\n\n                                def power(self, p):\n                                    print('compute power '+str(p))\n                                    return self.a**p\n                            "))
    self.shell.run_code('second = MyClass(5)')
    for object_name in {'first', 'second'}:
        self.shell.run_code(f'{object_name}.power(5)')
        with self.assertRaises(AttributeError):
            self.shell.run_code(f'{object_name}.cube()')
        with self.assertRaises(AttributeError):
            self.shell.run_code(f'{object_name}.square()')
        self.shell.run_code(f'{object_name}.b')
        self.shell.run_code(f'{object_name}.a')
        with self.assertRaises(AttributeError):
            self.shell.run_code(f'{object_name}.toto')