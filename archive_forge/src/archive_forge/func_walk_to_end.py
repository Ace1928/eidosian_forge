import re
from django.utils.functional import SimpleLazyObject
def walk_to_end(ch, input_iter):
    """
    The iterator is currently inside a capturing group. Walk to the close of
    this group, skipping over any nested groups and handling escaped
    parentheses correctly.
    """
    if ch == '(':
        nesting = 1
    else:
        nesting = 0
    for ch, escaped in input_iter:
        if escaped:
            continue
        elif ch == '(':
            nesting += 1
        elif ch == ')':
            if not nesting:
                return
            nesting -= 1