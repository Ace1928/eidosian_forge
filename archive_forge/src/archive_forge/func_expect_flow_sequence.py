from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_flow_sequence(self):
    ind = self.indents.seq_flow_align(self.best_sequence_indent, self.column)
    self.write_indicator(u' ' * ind + u'[', True, whitespace=True)
    self.increase_indent(flow=True, sequence=True)
    self.flow_context.append('[')
    self.state = self.expect_first_flow_sequence_item