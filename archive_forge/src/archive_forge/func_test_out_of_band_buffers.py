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
def test_out_of_band_buffers(self):
    if self.protocol < 5:
        pytest.skip('Need Pickle Protocol 5 or later')
    np = pytest.importorskip('numpy')

    class LocallyDefinedClass:
        data = np.zeros(10)
    data_instance = LocallyDefinedClass()
    buffers = []
    pickle_bytes = cloudpickle.dumps(data_instance, protocol=self.protocol, buffer_callback=buffers.append)
    assert len(buffers) == 1
    reconstructed = pickle.loads(pickle_bytes, buffers=buffers)
    np.testing.assert_allclose(reconstructed.data, data_instance.data)