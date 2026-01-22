import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def compile_repl_group(source, pattern):
    """Compiles a replacement template group reference."""
    source.expect('<')
    name = parse_name(source, True, True)
    source.expect('>')
    if name.isdigit():
        index = int(name)
        if not 0 <= index <= pattern.groups:
            raise error('invalid group reference', source.string, source.pos)
        return index
    try:
        return pattern.groupindex[name]
    except KeyError:
        raise IndexError('unknown group')