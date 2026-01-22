from __future__ import absolute_import
from ruamel.yaml.error import YAMLError
from ruamel.yaml.compat import nprint, DBG_NODE, dbg, string_types, nprintf  # NOQA
from ruamel.yaml.util import RegExp
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def generate_anchor(self, node):
    try:
        anchor = node.anchor.value
    except:
        anchor = None
    if anchor is None:
        self.last_anchor_id += 1
        return self.ANCHOR_TEMPLATE % self.last_anchor_id
    return anchor