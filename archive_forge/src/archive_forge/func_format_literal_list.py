import ast
from .qt import ClassFlag, qt_class_flags
def format_literal_list(l_node, enclosing='{'):
    """Formats a list/tuple of number/string literals as C++ initializer list"""
    result = enclosing
    for i, el in enumerate(l_node.elts):
        if i > 0:
            result += ', '
        result += format_literal(el)
    result += CLOSING[enclosing]
    return result