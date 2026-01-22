import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_formatter_skips_script_tag_for_html_documents(self):
    doc = '\n  <script type="text/javascript">\n   console.log("< < hey > > ");\n  </script>\n'
    encoded = BeautifulSoup(doc, 'html.parser').encode()
    assert b'< < hey > >' in encoded