from __future__ import print_function
import sys
from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.compat import Directive
from sphinx.util.nodes import nested_parse_with_titles
def document_formatters(self):
    from pygments.formatters import FORMATTERS
    out = []
    for classname, data in sorted(FORMATTERS.items(), key=lambda x: x[0]):
        module = data[0]
        mod = __import__(module, None, None, [classname])
        self.filenames.add(mod.__file__)
        cls = getattr(mod, classname)
        docstring = cls.__doc__
        if isinstance(docstring, bytes):
            docstring = docstring.decode('utf8')
        heading = cls.__name__
        out.append(FMTERDOC % (heading, ', '.join(data[2]) or 'None', ', '.join(data[3]).replace('*', '\\*') or 'None', docstring))
    return ''.join(out)