import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_special_string_containers(self):
    soup = self.soup('<style>Some CSS</style><script>Some Javascript</script>')
    assert isinstance(soup.style.string, Stylesheet)
    assert isinstance(soup.script.string, Script)
    soup = self.soup('<style><!--Some CSS--></style>')
    assert isinstance(soup.style.string, Stylesheet)
    assert soup.style.string == '<!--Some CSS-->'
    assert isinstance(soup.style.string, Stylesheet)