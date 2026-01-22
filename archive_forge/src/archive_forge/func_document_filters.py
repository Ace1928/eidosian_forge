from __future__ import print_function
import sys
from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.compat import Directive
from sphinx.util.nodes import nested_parse_with_titles
def document_filters(self):
    from pygments.filters import FILTERS
    out = []
    for name, cls in FILTERS.items():
        self.filenames.add(sys.modules[cls.__module__].__file__)
        docstring = cls.__doc__
        if isinstance(docstring, bytes):
            docstring = docstring.decode('utf8')
        out.append(FILTERDOC % (cls.__name__, name, docstring))
    return ''.join(out)