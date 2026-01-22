import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_nested_namespaces(self):
    doc = b'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n<parent xmlns="http://ns1/">\n<child xmlns="http://ns2/" xmlns:ns3="http://ns3/">\n<grandchild ns3:attr="value" xmlns="http://ns4/"/>\n</child>\n</parent>'
    soup = self.soup(doc)
    assert doc == soup.encode()