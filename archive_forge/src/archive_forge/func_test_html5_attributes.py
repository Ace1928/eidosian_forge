from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest
def test_html5_attributes(self):
    for input_element, output_unicode, output_element in (('&RightArrowLeftArrow;', '⇄', b'&rlarr;'), ('&models;', '⊧', b'&models;'), ('&Nfr;', '𝔑', b'&Nfr;'), ('&ngeqq;', '≧̸', b'&ngeqq;'), ('&not;', '¬', b'&not;'), ('&Not;', '⫬', b'&Not;'), ('&quot;', '"', b'"'), ('&there4;', '∴', b'&there4;'), ('&Therefore;', '∴', b'&there4;'), ('&therefore;', '∴', b'&there4;'), ('&fjlig;', 'fj', b'fj'), ('&sqcup;', '⊔', b'&sqcup;'), ('&sqcups;', '⊔︀', b'&sqcups;'), ('&apos;', "'", b"'"), ('&verbar;', '|', b'|')):
        markup = '<div>%s</div>' % input_element
        div = self.soup(markup).div
        without_element = div.encode()
        expect = b'<div>%s</div>' % output_unicode.encode('utf8')
        assert without_element == expect
        with_element = div.encode(formatter='html')
        expect = b'<div>%s</div>' % output_element
        assert with_element == expect