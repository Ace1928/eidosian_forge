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
def test_pickle_constructs_from_installed_packages_registered_for_pickling_by_value(self):
    pytest.importorskip('_cloudpickle_testpkg')
    for package_or_module in ['package', 'module']:
        if package_or_module == 'package':
            import _cloudpickle_testpkg as m
            f = m.package_function_with_global
            _original_global = m.global_variable
        elif package_or_module == 'module':
            import _cloudpickle_testpkg.mod as m
            f = m.module_function_with_global
            _original_global = m.global_variable
        try:
            with subprocess_worker(protocol=self.protocol) as w:
                assert w.run(lambda: f()) == _original_global
                register_pickle_by_value(m)
                assert w.run(lambda: f()) == _original_global
                m.global_variable = 'modified global'
                assert m.global_variable != _original_global
                assert w.run(lambda: f()) == 'modified global'
                unregister_pickle_by_value(m)
        finally:
            m.global_variable = _original_global
            if m.__name__ in list_registry_pickle_by_value():
                unregister_pickle_by_value(m)