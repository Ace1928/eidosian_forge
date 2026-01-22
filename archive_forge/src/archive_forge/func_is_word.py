import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def is_word(tok):
    return not tok.startswith('<')