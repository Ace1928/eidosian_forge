from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def increase_indent(self, flow=False, sequence=None, indentless=False):
    self.indents.append(self.indent, sequence)
    if self.indent is None:
        if flow:
            self.indent = self.requested_indent
        else:
            self.indent = 0
    elif not indentless:
        self.indent += self.best_sequence_indent if self.indents.last_seq() else self.best_map_indent