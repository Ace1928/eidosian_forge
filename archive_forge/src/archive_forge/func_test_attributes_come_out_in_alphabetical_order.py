import warnings
from bs4.element import (
from . import SoupTest
def test_attributes_come_out_in_alphabetical_order(self):
    markup = '<b a="1" z="5" m="3" f="2" y="4"></b>'
    self.assertSoupEquals(markup, '<b a="1" f="2" m="3" y="4" z="5"></b>')