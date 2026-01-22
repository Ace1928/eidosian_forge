import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_system_doctype(self):
    self.assertDoctypeHandled('foo SYSTEM "http://www.example.com/"')