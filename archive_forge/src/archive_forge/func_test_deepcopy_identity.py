import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_deepcopy_identity(self):
    copied = copy.deepcopy(self.tree)
    assert copied.decode() == self.tree.decode()