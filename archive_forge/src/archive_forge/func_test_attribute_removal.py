import unittest
from unittest import mock
import pytest
from ..pkg_info import cmp_pkg_version
def test_attribute_removal():
    for module_name, cls, attr in _filter(ATTRIBUTE_SCHEDULE):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        try:
            klass = getattr(module, cls)
        except AttributeError:
            continue
        assert not hasattr(klass, attr), f'Time to remove {module_name}.{cls}.{attr}'