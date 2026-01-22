import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_unsupported_pseudoclass(self):
    with pytest.raises(NotImplementedError):
        self.soup.select('a:no-such-pseudoclass')
    with pytest.raises(SelectorSyntaxError):
        self.soup.select('a:nth-of-type(a)')