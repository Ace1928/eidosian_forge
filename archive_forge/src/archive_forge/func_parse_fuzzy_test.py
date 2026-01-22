import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_fuzzy_test(source, info, case_flags):
    saved_pos = source.pos
    ch = source.get()
    if ch in SPECIAL_CHARS:
        if ch == '\\':
            return parse_escape(source, info, False)
        elif ch == '.':
            if info.flags & DOTALL:
                return AnyAll()
            elif info.flags & WORD:
                return AnyU()
            else:
                return Any()
        elif ch == '[':
            return parse_set(source, info)
        else:
            raise error('expected character set', source.string, saved_pos)
    elif ch:
        return Character(ord(ch), case_flags=case_flags)
    else:
        raise error('expected character set', source.string, saved_pos)