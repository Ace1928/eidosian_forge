from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_data(self, data):
    if self.ignore_aliases(data):
        self.alias_key = None
    else:
        self.alias_key = id(data)
    if self.alias_key is not None:
        if self.alias_key in self.represented_objects:
            node = self.represented_objects[self.alias_key]
            return node
        self.object_keeper.append(data)
    data_types = type(data).__mro__
    if data_types[0] in self.yaml_representers:
        node = self.yaml_representers[data_types[0]](self, data)
    else:
        for data_type in data_types:
            if data_type in self.yaml_multi_representers:
                node = self.yaml_multi_representers[data_type](self, data)
                break
        else:
            if None in self.yaml_multi_representers:
                node = self.yaml_multi_representers[None](self, data)
            elif None in self.yaml_representers:
                node = self.yaml_representers[None](self, data)
            else:
                node = ScalarNode(None, str(data))
    return node