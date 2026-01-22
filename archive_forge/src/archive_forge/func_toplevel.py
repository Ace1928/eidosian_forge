from __future__ import unicode_literals
from pybtex.style import FormattedEntry, FormattedBibliography
from pybtex.style.template import node, join
from pybtex.richtext import Symbol
from pybtex.plugin import Plugin, find_plugin
@node
def toplevel(children, data):
    return join(sep=Symbol('newblock'))[children].format_data(data)