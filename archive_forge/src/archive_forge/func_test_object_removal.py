import unittest
from unittest import mock
import pytest
from ..pkg_info import cmp_pkg_version
def test_object_removal():
    for module_name, obj in _filter(OBJECT_SCHEDULE):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        assert not hasattr(module, obj), f'Time to remove {module_name}.{obj}'