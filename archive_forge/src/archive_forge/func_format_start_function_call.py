import ast
from .qt import ClassFlag, qt_class_flags
def format_start_function_call(call_node):
    """Format a call of a free or member function"""
    return format_reference(call_node.func) + '('