from lxml import etree
import sys
import re
import doctest
def get_parser(self, want, got, optionflags):
    parser = None
    if NOPARSE_MARKUP & optionflags:
        return None
    if PARSE_HTML & optionflags:
        parser = html_fromstring
    elif PARSE_XML & optionflags:
        parser = etree.XML
    elif want.strip().lower().startswith('<html') and got.strip().startswith('<html'):
        parser = html_fromstring
    elif self._looks_like_markup(want) and self._looks_like_markup(got):
        parser = self.get_default_parser()
    return parser