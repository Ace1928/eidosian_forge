import ast
from .qt import ClassFlag, qt_class_flags
def format_name_constant(node):
    """Format a ast.NameConstant."""
    if node.value is None:
        return 'nullptr'
    return 'true' if node.value else 'false'