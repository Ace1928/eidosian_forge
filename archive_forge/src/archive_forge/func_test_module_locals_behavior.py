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
def test_module_locals_behavior(self):
    pickled_func_path = os.path.join(self.tmpdir, 'local_func_g.pkl')
    child_process_script = '\n        from srsly.cloudpickle.compat import pickle\n        import gc\n        with open("{pickled_func_path}", \'rb\') as f:\n            func = pickle.load(f)\n\n        assert func(range(10)) == 45\n        '
    child_process_script = child_process_script.format(pickled_func_path=_escape(pickled_func_path))
    try:
        from srsly.tests.cloudpickle.testutils import make_local_function
        g = make_local_function()
        with open(pickled_func_path, 'wb') as f:
            cloudpickle.dump(g, f, protocol=self.protocol)
        assert_run_python_script(textwrap.dedent(child_process_script))
    finally:
        os.unlink(pickled_func_path)