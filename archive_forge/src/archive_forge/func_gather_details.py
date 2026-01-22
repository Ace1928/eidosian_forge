import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
def gather_details(source_dict, target_dict):
    """Merge the details from ``source_dict`` into ``target_dict``.

    ``gather_details`` evaluates all details in ``source_dict``. Do not use it
    if the details are not ready to be evaluated.

    :param source_dict: A dictionary of details will be gathered.
    :param target_dict: A dictionary into which details will be gathered.
    """
    for name, content_object in source_dict.items():
        new_name = name
        disambiguator = itertools.count(1)
        while new_name in target_dict:
            new_name = '%s-%d' % (name, next(disambiguator))
        name = new_name
        target_dict[name] = _copy_content(content_object)