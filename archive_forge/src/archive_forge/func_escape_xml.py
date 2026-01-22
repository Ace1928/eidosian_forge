from __future__ import absolute_import, unicode_literals
import re
import sys
def escape_xml(s):
    if s is None:
        return ''
    if re.search(reXmlSpecial, s):
        return re.sub(reXmlSpecial, lambda m: replace_unsafe_char(m.group()), s)
    else:
        return s