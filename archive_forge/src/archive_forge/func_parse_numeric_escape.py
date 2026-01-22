import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_numeric_escape(source, info, ch, in_set):
    """Parses a numeric escape sequence."""
    if in_set or ch == '0':
        return parse_octal_escape(source, info, [ch], in_set)
    digits = ch
    saved_pos = source.pos
    ch = source.get()
    if ch in DIGITS:
        digits += ch
        saved_pos = source.pos
        ch = source.get()
        if is_octal(digits) and ch in OCT_DIGITS:
            encoding = info.flags & _ALL_ENCODINGS
            if encoding == ASCII or encoding == LOCALE:
                octal_mask = 255
            else:
                octal_mask = 511
            value = int(digits + ch, 8) & octal_mask
            return make_character(info, value)
    source.pos = saved_pos
    if info.is_open_group(digits):
        raise error('cannot refer to an open group', source.string, source.pos)
    return make_ref_group(info, digits, source.pos)