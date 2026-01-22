from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
def define_name(self, name, node):
    try:
        name_obj = self.names[name]
    except KeyError:
        name_obj = self.names[name] = Name(name)
    name_obj.define(node)
    return name_obj