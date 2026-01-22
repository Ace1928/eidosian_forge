from __future__ import absolute_import, print_function
import warnings
from ruamel.yaml.error import MarkedYAMLError, ReusedAnchorWarning
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def check_end_doc_comment(self, end_event, node):
    if end_event.comment and end_event.comment[1]:
        if node.comment is None:
            node.comment = [None, None]
        assert not isinstance(node, ScalarEvent)
        node.comment.append(end_event.comment[1])
        end_event.comment[1] = None