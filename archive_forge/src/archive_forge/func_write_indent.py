from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_indent(self):
    indent = self.indent or 0
    if not self.indention or self.column > indent or (self.column == indent and (not self.whitespace)):
        if bool(self.no_newline):
            self.no_newline = False
        else:
            self.write_line_break()
    if self.column < indent:
        self.whitespace = True
        data = u' ' * (indent - self.column)
        self.column = indent
        if self.encoding:
            data = data.encode(self.encoding)
        self.stream.write(data)