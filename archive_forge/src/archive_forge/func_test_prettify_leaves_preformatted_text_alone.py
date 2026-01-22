import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_prettify_leaves_preformatted_text_alone(self):
    soup = self.soup('<div>  foo  <pre>  \tbar\n  \n  </pre>  baz  <textarea> eee\nfff\t</textarea></div>')
    assert '<div>\n foo\n <pre>  \tbar\n  \n  </pre>\n baz\n <textarea> eee\nfff\t</textarea>\n</div>\n' == soup.div.prettify()