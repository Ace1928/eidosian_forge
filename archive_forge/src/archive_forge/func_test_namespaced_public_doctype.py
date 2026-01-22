import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_namespaced_public_doctype(self):
    self.assertDoctypeHandled('xsl:stylesheet PUBLIC "htmlent.dtd"')