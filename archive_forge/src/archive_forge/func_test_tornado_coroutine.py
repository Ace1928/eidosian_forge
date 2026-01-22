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
@pytest.mark.skipif(tornado is None, reason='test needs Tornado installed')
def test_tornado_coroutine(self):
    from tornado import gen, ioloop

    @gen.coroutine
    def f(x, y):
        yield gen.sleep(x)
        raise gen.Return(y + 1)

    @gen.coroutine
    def g(y):
        res = (yield f(0.01, y))
        raise gen.Return(res + 1)
    data = cloudpickle.dumps([g, g], protocol=self.protocol)
    f = g = None
    g2, g3 = pickle.loads(data)
    self.assertTrue(g2 is g3)
    loop = ioloop.IOLoop.current()
    res = loop.run_sync(functools.partial(g2, 5))
    self.assertEqual(res, 7)