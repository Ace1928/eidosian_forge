from __future__ import absolute_import, print_function
import warnings
from ruamel.yaml.error import MarkedYAMLError, ReusedAnchorWarning
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def get_single_node(self):
    self.parser.get_event()
    document = None
    if not self.parser.check_event(StreamEndEvent):
        document = self.compose_document()
    if not self.parser.check_event(StreamEndEvent):
        event = self.parser.get_event()
        raise ComposerError('expected a single document in the stream', document.start_mark, 'but found another document', event.start_mark)
    self.parser.get_event()
    return document