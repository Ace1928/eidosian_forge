import re
import html
def replace_subscript(s: str, subscript=True) -> str:
    target = '~'
    rdict = subscript_dict
    if not subscript:
        target = '^'
        rdict = superscript_dict
    replaced = []
    inside = False
    for char in s:
        if char == target:
            inside = not inside
        elif not inside:
            replaced += [char]
        elif char in rdict:
            replaced += [rdict[char]]
        else:
            replaced += [char]
    return ''.join(replaced)