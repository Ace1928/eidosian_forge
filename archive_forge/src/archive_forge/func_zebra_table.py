import re
import html
from paste.util import PySourceColor
def zebra_table(self, title, rows, table_class='variables'):
    if isinstance(rows, dict):
        rows = rows.items()
        rows = sorted(rows)
    table = ['<table class="%s">' % table_class, '<tr class="header"><th colspan="2">%s</th></tr>' % self.quote(title)]
    odd = False
    for name, value in rows:
        try:
            value = repr(value)
        except Exception as e:
            value = 'Cannot print: %s' % e
        odd = not odd
        table.append('<tr class="%s"><td>%s</td>' % (odd and 'odd' or 'even', self.quote(name)))
        table.append('<td><tt>%s</tt></td></tr>' % make_wrappable(self.quote(truncate(value))))
    table.append('</table>')
    return '\n'.join(table)