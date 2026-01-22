from lxml import etree
import sys
import re
import doctest
def tag_compare(self, want, got):
    if want == 'any':
        return True
    if not isinstance(want, (str, bytes)) or not isinstance(got, (str, bytes)):
        return want == got
    want = want or ''
    got = got or ''
    if want.startswith('{...}'):
        return want.split('}')[-1] == got.split('}')[-1]
    else:
        return want == got