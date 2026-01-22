import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_multi_class_support(self):
    for selector in ('.class1', 'p.class1', '.class2', 'p.class2', '.class3', 'p.class3', 'html p.class2', 'div#inner .class2'):
        self.assert_selects(selector, ['pmulti'])