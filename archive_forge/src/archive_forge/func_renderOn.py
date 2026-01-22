import formatter
import string
from types import *
import htmllib
import piddle
def renderOn(self, aPiddleCanvas):
    """draw the text with aPiddleCanvas
            jjk  02/01/00"""
    writer = _HtmlPiddleWriter(self, aPiddleCanvas)
    fmt = formatter.AbstractFormatter(writer)
    parser = _HtmlParser(fmt)
    parser.feed(self.html)
    parser.close()