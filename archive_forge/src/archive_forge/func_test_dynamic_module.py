import _collections_abc
import abc
import collections
import base64
import functools
import io
import itertools
import logging
import math
import multiprocessing
from operator import itemgetter, attrgetter
import pickletools
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import weakref
import os
import enum
import typing
from functools import wraps
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
from srsly.cloudpickle import register_pickle_by_value
from srsly.cloudpickle import unregister_pickle_by_value
from srsly.cloudpickle import list_registry_pickle_by_value
from srsly.cloudpickle.cloudpickle import _should_pickle_by_reference
from srsly.cloudpickle.cloudpickle import _make_empty_cell, cell_set
from srsly.cloudpickle.cloudpickle import _extract_class_dict, _whichmodule
from srsly.cloudpickle.cloudpickle import _lookup_module_and_qualname
from .testutils import subprocess_pickle_echo
from .testutils import subprocess_pickle_string
from .testutils import assert_run_python_script
from .testutils import subprocess_worker
def test_dynamic_module(self):
    mod = types.ModuleType('mod')
    code = '\n        x = 1\n        def f(y):\n            return x + y\n\n        class Foo:\n            def method(self, x):\n                return f(x)\n        '
    exec(textwrap.dedent(code), mod.__dict__)
    mod2 = pickle_depickle(mod, protocol=self.protocol)
    self.assertEqual(mod.x, mod2.x)
    self.assertEqual(mod.f(5), mod2.f(5))
    self.assertEqual(mod.Foo().method(5), mod2.Foo().method(5))
    if platform.python_implementation() != 'PyPy':
        mod3 = subprocess_pickle_echo(mod, protocol=self.protocol)
        self.assertEqual(mod.x, mod3.x)
        self.assertEqual(mod.f(5), mod3.f(5))
        self.assertEqual(mod.Foo().method(5), mod3.Foo().method(5))
    mod1, mod2 = pickle_depickle([mod, mod])
    self.assertEqual(id(mod1), id(mod2))
    try:
        sys.modules['mod'] = mod
        depickled_f = pickle_depickle(mod.f, protocol=self.protocol)
        self.assertEqual(mod.f(5), depickled_f(5))
    finally:
        sys.modules.pop('mod', None)