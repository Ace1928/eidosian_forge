from html import escape
from io import StringIO
from incremental import Version
from twisted.python import log
from twisted.python.deprecate import deprecated
@deprecated(Version('Twisted', 15, 3, 0), replacement='twisted.web.template')
def linkList(lst):
    io = StringIO()
    io.write('<ul>\n')
    for hr, el in lst:
        io.write(f'<li> <a href="{hr}">{el}</a></li>\n')
    io.write('</ul>')
    return io.getvalue()