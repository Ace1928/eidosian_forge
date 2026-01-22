import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def is_start_tag(tok):
    return tok.startswith('<') and (not tok.startswith('</'))