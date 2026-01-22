import re
import _markupbase
from html import unescape
def get_starttag_text(self):
    """Return full source of start tag: '<...>'."""
    return self.__starttag_text