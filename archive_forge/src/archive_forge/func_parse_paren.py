import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_paren(source, info):
    """Parses a parenthesised subpattern or a flag. Returns FLAGS if it's an
    inline flag.
    """
    saved_pos = source.pos
    ch = source.get(True)
    if ch == '?':
        saved_pos_2 = source.pos
        ch = source.get(True)
        if ch == '<':
            saved_pos_3 = source.pos
            ch = source.get()
            if ch in ('=', '!'):
                return parse_lookaround(source, info, True, ch == '=')
            source.pos = saved_pos_3
            name = parse_name(source)
            group = info.open_group(name)
            source.expect('>')
            saved_flags = info.flags
            try:
                subpattern = _parse_pattern(source, info)
                source.expect(')')
            finally:
                info.flags = saved_flags
                source.ignore_space = bool(info.flags & VERBOSE)
            info.close_group()
            return Group(info, group, subpattern)
        if ch in ('=', '!'):
            return parse_lookaround(source, info, False, ch == '=')
        if ch == 'P':
            return parse_extension(source, info)
        if ch == '#':
            return parse_comment(source)
        if ch == '(':
            return parse_conditional(source, info)
        if ch == '>':
            return parse_atomic(source, info)
        if ch == '|':
            return parse_common(source, info)
        if ch == 'R' or '0' <= ch <= '9':
            return parse_call_group(source, info, ch, saved_pos_2)
        if ch == '&':
            return parse_call_named_group(source, info, saved_pos_2)
        source.pos = saved_pos_2
        return parse_flags_subpattern(source, info)
    if ch == '*':
        saved_pos_2 = source.pos
        word = source.get_while(set(')>'), include=False)
        if word[:1].isalpha():
            verb = VERBS.get(word)
            if not verb:
                raise error('unknown verb', source.string, saved_pos_2)
            source.expect(')')
            return verb
    source.pos = saved_pos
    group = info.open_group()
    saved_flags = info.flags
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    info.close_group()
    return Group(info, group, subpattern)