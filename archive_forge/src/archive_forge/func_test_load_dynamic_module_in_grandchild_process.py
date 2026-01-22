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
def test_load_dynamic_module_in_grandchild_process(self):
    mod = types.ModuleType('mod')
    code = '\n        x = 1\n        '
    exec(textwrap.dedent(code), mod.__dict__)
    parent_process_module_file = os.path.join(self.tmpdir, 'dynamic_module_from_parent_process.pkl')
    child_process_module_file = os.path.join(self.tmpdir, 'dynamic_module_from_child_process.pkl')
    child_process_script = "\n            from srsly.cloudpickle.compat import pickle\n            import textwrap\n\n            import srsly.cloudpickle as cloudpickle\n            from srsly.tests.cloudpickle.testutils import assert_run_python_script\n\n\n            child_of_child_process_script = {child_of_child_process_script}\n\n            with open('{parent_process_module_file}', 'rb') as f:\n                mod = pickle.load(f)\n\n            with open('{child_process_module_file}', 'wb') as f:\n                cloudpickle.dump(mod, f, protocol={protocol})\n\n            assert_run_python_script(textwrap.dedent(child_of_child_process_script))\n            "
    child_of_child_process_script = " '''\n                from srsly.cloudpickle.compat import pickle\n                with open('{child_process_module_file}','rb') as fid:\n                    mod = pickle.load(fid)\n                ''' "
    child_of_child_process_script = child_of_child_process_script.format(child_process_module_file=child_process_module_file)
    child_process_script = child_process_script.format(parent_process_module_file=_escape(parent_process_module_file), child_process_module_file=_escape(child_process_module_file), child_of_child_process_script=_escape(child_of_child_process_script), protocol=self.protocol)
    try:
        with open(parent_process_module_file, 'wb') as fid:
            cloudpickle.dump(mod, fid, protocol=self.protocol)
        assert_run_python_script(textwrap.dedent(child_process_script))
    finally:
        if os.path.exists(parent_process_module_file):
            os.unlink(parent_process_module_file)
        if os.path.exists(child_process_module_file):
            os.unlink(child_process_module_file)