from __future__ import absolute_import, print_function
import warnings
from ruamel.yaml.error import MarkedYAMLError, ReusedAnchorWarning
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def compose_scalar_node(self, anchor):
    event = self.parser.get_event()
    tag = event.tag
    if tag is None or tag == u'!':
        tag = self.resolver.resolve(ScalarNode, event.value, event.implicit)
    node = ScalarNode(tag, event.value, event.start_mark, event.end_mark, style=event.style, comment=event.comment, anchor=anchor)
    if anchor is not None:
        self.anchors[anchor] = node
    return node