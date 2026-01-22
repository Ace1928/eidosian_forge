import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_formatter_processes_script_tag_for_xml_documents(self):
    doc = '\n  <script type="text/javascript">\n  </script>\n'
    soup = BeautifulSoup(doc, 'lxml-xml')
    soup.script.string = 'console.log("< < hey > > ");'
    encoded = soup.encode()
    assert b'&lt; &lt; hey &gt; &gt;' in encoded