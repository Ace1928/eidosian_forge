import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_key_val_pair(self, node, value):
    return self.visit(node['children'][0], value)