from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_mapping(self, tag, mapping, flow_style=None):
    value = []
    node = MappingNode(tag, value, flow_style=flow_style)
    if self.alias_key is not None:
        self.represented_objects[self.alias_key] = node
    best_style = True
    if hasattr(mapping, 'items'):
        mapping = list(mapping.items())
        if self.sort_keys:
            try:
                mapping = sorted(mapping)
            except TypeError:
                pass
    for item_key, item_value in mapping:
        node_key = self.represent_data(item_key)
        node_value = self.represent_data(item_value)
        if not (isinstance(node_key, ScalarNode) and (not node_key.style)):
            best_style = False
        if not (isinstance(node_value, ScalarNode) and (not node_value.style)):
            best_style = False
        value.append((node_key, node_value))
    if flow_style is None:
        if self.default_flow_style is not None:
            node.flow_style = self.default_flow_style
        else:
            node.flow_style = best_style
    return node