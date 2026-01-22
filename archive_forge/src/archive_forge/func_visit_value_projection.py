import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_value_projection(self, node, value):
    base = self.visit(node['children'][0], value)
    try:
        base = base.values()
    except AttributeError:
        return None
    collected = []
    for element in base:
        current = self.visit(node['children'][1], element)
        if current is not None:
            collected.append(current)
    return collected