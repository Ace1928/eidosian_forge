from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest
def test_rejected_input(self):
    bad_markup = [b'\n<![\xff\xfe\xfe\xcd\x00', b'<![n\x00', b'<![UNKNOWN[]]>']
    for markup in bad_markup:
        with pytest.raises(ParserRejectedMarkup):
            soup = self.soup(markup)