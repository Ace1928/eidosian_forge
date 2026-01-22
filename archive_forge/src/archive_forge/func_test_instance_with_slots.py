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
def test_instance_with_slots(self):
    for slots in [['registered_attribute'], 'registered_attribute']:

        class ClassWithSlots:
            __slots__ = slots

            def __init__(self):
                self.registered_attribute = 42
        initial_obj = ClassWithSlots()
        depickled_obj = pickle_depickle(initial_obj, protocol=self.protocol)
        for obj in [initial_obj, depickled_obj]:
            self.assertEqual(obj.registered_attribute, 42)
            with pytest.raises(AttributeError):
                obj.non_registered_attribute = 1

        class SubclassWithSlots(ClassWithSlots):

            def __init__(self):
                self.unregistered_attribute = 1
        obj = SubclassWithSlots()
        s = cloudpickle.dumps(obj, protocol=self.protocol)
        del SubclassWithSlots
        depickled_obj = cloudpickle.loads(s)
        assert depickled_obj.unregistered_attribute == 1