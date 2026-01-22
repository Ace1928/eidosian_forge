from lxml import etree
import sys
import re
import doctest
def format_end_tag(self, el):
    if isinstance(el, etree.CommentBase):
        return '-->'
    return '</%s>' % el.tag