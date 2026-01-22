import re
import html
from paste.util import PySourceColor
def format_source(self, source_line):
    return '&nbsp;&nbsp;<code class="source">%s</code>' % self.quote(source_line.strip())