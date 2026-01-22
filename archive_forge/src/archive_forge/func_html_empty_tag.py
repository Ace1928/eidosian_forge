from lxml import etree
import sys
import re
import doctest
def html_empty_tag(self, el, html=True):
    if not html:
        return False
    if el.tag not in self.empty_tags:
        return False
    if el.text or len(el):
        return False
    return True