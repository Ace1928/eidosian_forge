from lxml import etree
import sys
import re
import doctest
def collect_diff_text(self, want, got, strip=True):
    if self.text_compare(want, got, strip):
        if not got:
            return ''
        return self.format_text(got, strip)
    text = '%s (got: %s)' % (want, got)
    return self.format_text(text, strip)