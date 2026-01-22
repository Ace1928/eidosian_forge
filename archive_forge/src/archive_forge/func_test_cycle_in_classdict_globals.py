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
def test_cycle_in_classdict_globals(self):

    class C:

        def it_works(self):
            return 'woohoo!'
    C.C_again = C
    C.instance_of_C = C()
    depickled_C = pickle_depickle(C, protocol=self.protocol)
    depickled_instance = pickle_depickle(C())
    self.assertEqual(depickled_C().it_works(), 'woohoo!')
    self.assertEqual(depickled_C.C_again().it_works(), 'woohoo!')
    self.assertEqual(depickled_C.instance_of_C.it_works(), 'woohoo!')
    self.assertEqual(depickled_instance.it_works(), 'woohoo!')