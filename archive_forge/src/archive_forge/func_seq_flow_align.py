from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def seq_flow_align(self, seq_indent, column):
    if len(self.values) < 2 or not self.values[-1][1]:
        return 0
    base = self.values[-1][0] if self.values[-1][0] is not None else 0
    return base + seq_indent - column - 1