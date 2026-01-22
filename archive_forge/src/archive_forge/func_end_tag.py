import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def end_tag(el):
    """ The text representation of an end tag for a tag.  Includes
    trailing whitespace when appropriate.  """
    if el.tail and start_whitespace_re.search(el.tail):
        extra = ' '
    else:
        extra = ''
    return '</%s>%s' % (el.tag, extra)