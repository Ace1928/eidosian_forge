from lxml import etree
import sys
import re
import doctest
def format_doc(self, doc, html, indent, prefix=''):
    parts = []
    if not len(doc):
        parts.append(' ' * indent)
        parts.append(prefix)
        parts.append(self.format_tag(doc))
        if not self.html_empty_tag(doc, html):
            if strip(doc.text):
                parts.append(self.format_text(doc.text))
            parts.append(self.format_end_tag(doc))
        if strip(doc.tail):
            parts.append(self.format_text(doc.tail))
        parts.append('\n')
        return ''.join(parts)
    parts.append(' ' * indent)
    parts.append(prefix)
    parts.append(self.format_tag(doc))
    if not self.html_empty_tag(doc, html):
        parts.append('\n')
        if strip(doc.text):
            parts.append(' ' * indent)
            parts.append(self.format_text(doc.text))
            parts.append('\n')
        for el in doc:
            parts.append(self.format_doc(el, html, indent + 2))
        parts.append(' ' * indent)
        parts.append(self.format_end_tag(doc))
        parts.append('\n')
    if strip(doc.tail):
        parts.append(' ' * indent)
        parts.append(self.format_text(doc.tail))
        parts.append('\n')
    return ''.join(parts)