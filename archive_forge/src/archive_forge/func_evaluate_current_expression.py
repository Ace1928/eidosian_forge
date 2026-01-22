import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
def evaluate_current_expression(cursor_offset: int, line: str, namespace: Optional[Dict[str, Any]]=None) -> Any:
    """
    Return evaluated expression to the right of the dot of current attribute.

    Only evaluates builtin objects, and do any attribute lookup.
    """
    temp_line = line[:cursor_offset] + 'xxx' + line[cursor_offset:]
    temp_cursor = cursor_offset + 3
    temp_attribute = line_properties.current_expression_attribute(temp_cursor, temp_line)
    if temp_attribute is None:
        raise EvaluationError('No current attribute')
    attr_before_cursor = temp_line[temp_attribute.start:temp_cursor]

    def parse_trees(cursor_offset, line):
        for i in range(cursor_offset - 1, -1, -1):
            try:
                tree = ast.parse(line[i:cursor_offset])
                yield tree
            except SyntaxError:
                continue
    largest_ast = None
    for tree in parse_trees(temp_cursor, temp_line):
        attribute_access = find_attribute_with_name(tree, attr_before_cursor)
        if attribute_access:
            largest_ast = attribute_access.value
    if largest_ast is None:
        raise EvaluationError('Corresponding ASTs to right of cursor are invalid')
    try:
        return simple_eval(largest_ast, namespace)
    except ValueError:
        raise EvaluationError('Could not safely evaluate')