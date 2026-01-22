import re
import html
from paste.util import PySourceColor
def format_source_line(self, filename, frame):
    name = self.quote(frame.name or '?')
    return 'Module <span class="module" title="%s">%s</span>:<b>%s</b> in <code>%s</code>' % (filename, frame.modname or '?', frame.lineno or '?', name)
    return 'File %r, line %s in <tt>%s</tt>' % (filename, frame.lineno, name)