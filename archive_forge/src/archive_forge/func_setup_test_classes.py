from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def setup_test_classes():
    for test_class in test_classes:
        add_markers = set()
        if getattr(test_class.cls, '__backend__', False) or getattr(test_class.cls, '__only_on__', False):
            add_markers = {'backend'}
        elif getattr(test_class.cls, '__sparse_backend__', False):
            add_markers = {'sparse_backend'}
        else:
            add_markers = frozenset()
        existing_markers = {mark.name for mark in test_class.iter_markers()}
        add_markers = add_markers - existing_markers
        all_markers = existing_markers.union(add_markers)
        for marker in add_markers:
            test_class.add_marker(marker)
        for sub_cls in plugin_base.generate_sub_tests(test_class.cls, test_class.module, all_markers):
            if sub_cls is not test_class.cls:
                per_cls_dict = rebuilt_items[test_class.cls]
                module = test_class.getparent(pytest.Module)
                new_cls = pytest.Class.from_parent(name=sub_cls.__name__, parent=module)
                for marker in add_markers:
                    new_cls.add_marker(marker)
                for fn in collect(new_cls):
                    per_cls_dict[fn.name].append(fn)