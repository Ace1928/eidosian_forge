from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def choose_scalar_style(self):
    if self.analysis is None:
        self.analysis = self.analyze_scalar(self.event.value)
    if self.event.style == '"' or self.canonical:
        return '"'
    if (not self.event.style or self.event.style == '?') and (self.event.implicit[0] or not self.event.implicit[2]):
        if not (self.simple_key_context and (self.analysis.empty or self.analysis.multiline)) and (self.flow_level and self.analysis.allow_flow_plain or (not self.flow_level and self.analysis.allow_block_plain)):
            return ''
    self.analysis.allow_block = True
    if self.event.style and self.event.style in '|>':
        if not self.flow_level and (not self.simple_key_context) and self.analysis.allow_block:
            return self.event.style
    if not self.event.style and self.analysis.allow_double_quoted:
        if "'" in self.event.value or '\n' in self.event.value:
            return '"'
    if not self.event.style or self.event.style == "'":
        if self.analysis.allow_single_quoted and (not (self.simple_key_context and self.analysis.multiline)):
            return "'"
    return '"'