import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_flatten(self, node, value):
    base = self.visit(node['children'][0], value)
    if not isinstance(base, list):
        return None
    merged_list = []
    for element in base:
        if isinstance(element, list):
            merged_list.extend(element)
        else:
            merged_list.append(element)
    return merged_list