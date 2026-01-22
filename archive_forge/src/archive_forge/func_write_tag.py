import re
from html import escape
from html.entities import name2codepoint
from html.parser import HTMLParser
def write_tag(self, tag, attrs, startend=False):
    attr_text = ''.join((' %s="%s"' % (n, html_quote(v)) for n, v in attrs if not n.startswith('form:')))
    if startend:
        attr_text += ' /'
    self.write_text('<%s%s>' % (tag, attr_text))