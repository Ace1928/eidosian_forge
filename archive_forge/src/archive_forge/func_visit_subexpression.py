import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_subexpression(self, node, value):
    result = value
    for node in node['children']:
        result = self.visit(node, result)
    return result