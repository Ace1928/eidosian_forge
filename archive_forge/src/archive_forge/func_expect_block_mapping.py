from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_block_mapping(self):
    if not self.mapping_context and (not (self.compact_seq_map or self.column == 0)):
        self.write_line_break()
    self.increase_indent(flow=False, sequence=False)
    self.state = self.expect_first_block_mapping_key