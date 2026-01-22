from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def check_simple_key(self):
    length = 0
    if isinstance(self.event, NodeEvent) and self.event.anchor is not None:
        if self.prepared_anchor is None:
            self.prepared_anchor = self.prepare_anchor(self.event.anchor)
        length += len(self.prepared_anchor)
    if isinstance(self.event, (ScalarEvent, CollectionStartEvent)) and self.event.tag is not None:
        if self.prepared_tag is None:
            self.prepared_tag = self.prepare_tag(self.event.tag)
        length += len(self.prepared_tag)
    if isinstance(self.event, ScalarEvent):
        if self.analysis is None:
            self.analysis = self.analyze_scalar(self.event.value)
        length += len(self.analysis.scalar)
    return length < self.MAX_SIMPLE_KEY_LENGTH and (isinstance(self.event, AliasEvent) or (isinstance(self.event, SequenceStartEvent) and self.event.flow_style is True) or (isinstance(self.event, MappingStartEvent) and self.event.flow_style is True) or (isinstance(self.event, ScalarEvent) and (not self.analysis.empty) and (not self.analysis.multiline)) or self.check_empty_sequence() or self.check_empty_mapping())