import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_not_expression(self, node, value):
    original_result = self.visit(node['children'][0], value)
    if type(original_result) is int and original_result == 0:
        return False
    return not original_result