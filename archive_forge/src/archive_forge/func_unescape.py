import re
from lxml import etree, html
def unescape(string):
    if not string:
        return ''

    def unescape_entity(m):
        try:
            return unichr(name2codepoint[m.group(1)])
        except KeyError:
            return m.group(0)
    return handle_entities(unescape_entity, string)