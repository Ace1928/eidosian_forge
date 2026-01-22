from __future__ import print_function
import sys
from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.compat import Directive
from sphinx.util.nodes import nested_parse_with_titles
def document_lexers(self):
    from pygments.lexers._mapping import LEXERS
    out = []
    modules = {}
    moduledocstrings = {}
    for classname, data in sorted(LEXERS.items(), key=lambda x: x[0]):
        module = data[0]
        mod = __import__(module, None, None, [classname])
        self.filenames.add(mod.__file__)
        cls = getattr(mod, classname)
        if not cls.__doc__:
            print('Warning: %s does not have a docstring.' % classname)
        docstring = cls.__doc__
        if isinstance(docstring, bytes):
            docstring = docstring.decode('utf8')
        modules.setdefault(module, []).append((classname, ', '.join(data[2]) or 'None', ', '.join(data[3]).replace('*', '\\*').replace('_', '\\') or 'None', ', '.join(data[4]) or 'None', docstring))
        if module not in moduledocstrings:
            moddoc = mod.__doc__
            if isinstance(moddoc, bytes):
                moddoc = moddoc.decode('utf8')
            moduledocstrings[module] = moddoc
    for module, lexers in sorted(modules.items(), key=lambda x: x[0]):
        if moduledocstrings[module] is None:
            raise Exception('Missing docstring for %s' % (module,))
        heading = moduledocstrings[module].splitlines()[4].strip().rstrip('.')
        out.append(MODULEDOC % (module, heading, '-' * len(heading)))
        for data in lexers:
            out.append(LEXERDOC % data)
    return ''.join(out)