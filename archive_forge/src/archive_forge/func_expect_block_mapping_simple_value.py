from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def expect_block_mapping_simple_value(self):
    if getattr(self.event, 'style', None) != '?':
        if self.indent == 0 and self.top_level_colon_align is not None:
            c = u' ' * (self.top_level_colon_align - self.column) + self.colon
        else:
            c = self.prefixed_colon
        self.write_indicator(c, False)
    self.states.append(self.expect_block_mapping_key)
    self.expect_node(mapping=True)