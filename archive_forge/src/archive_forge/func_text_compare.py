from lxml import etree
import sys
import re
import doctest
def text_compare(self, want, got, strip):
    want = want or ''
    got = got or ''
    if strip:
        want = norm_whitespace(want).strip()
        got = norm_whitespace(got).strip()
    want = '^%s$' % re.escape(want)
    want = want.replace('\\.\\.\\.', '.*')
    if re.search(want, got):
        return True
    else:
        return False