import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_slice(self, node, value):
    if not isinstance(value, list):
        return None
    s = slice(*node['children'])
    return value[s]