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
@pytest.mark.skipif(platform.machine() == 'aarch64' and sys.version_info[:2] >= (3, 10) or platform.python_implementation() == 'PyPy' or (sys.version_info[:2] == (3, 10) and sys.version_info >= (3, 10, 8)) or (sys.version_info[:2] >= (3, 11)), reason='Fails on aarch64 + python 3.10+ in cibuildwheel, currently unable to replicate failure elsewhere; fails sometimes for pypy on conda-forge; fails for python 3.10.8+ and 3.11+')
def test_builtin_classmethod(self):
    obj = 1.5
    bound_clsmethod = obj.fromhex
    unbound_clsmethod = type(obj).fromhex
    clsdict_clsmethod = type(obj).__dict__['fromhex']
    depickled_bound_meth = pickle_depickle(bound_clsmethod, protocol=self.protocol)
    depickled_unbound_meth = pickle_depickle(unbound_clsmethod, protocol=self.protocol)
    depickled_clsdict_meth = pickle_depickle(clsdict_clsmethod, protocol=self.protocol)
    arg = '0x1'
    assert depickled_bound_meth(arg) == bound_clsmethod(arg)
    assert depickled_unbound_meth(arg) == unbound_clsmethod(arg)
    if platform.python_implementation() == 'CPython':
        assert depickled_clsdict_meth(arg) == clsdict_clsmethod(float, arg)
    if platform.python_implementation() == 'PyPy':
        assert type(depickled_clsdict_meth) == type(clsdict_clsmethod)
        assert depickled_clsdict_meth.__func__(float, arg) == clsdict_clsmethod.__func__(float, arg)