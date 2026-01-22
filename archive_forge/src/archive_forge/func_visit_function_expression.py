import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_function_expression(self, node, value):
    resolved_args = []
    for child in node['children']:
        current = self.visit(child, value)
        resolved_args.append(current)
    return self._functions.call_function(node['value'], resolved_args)