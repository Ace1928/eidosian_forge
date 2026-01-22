from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
def tokenize_lines(lines: Iterable[str], *, version_info: PythonVersionInfo, indents: List[int]=None, start_pos: Tuple[int, int]=(1, 0), is_first_token=True) -> Iterator[PythonToken]:
    """
    A heavily modified Python standard library tokenizer.

    Additionally to the default information, yields also the prefix of each
    token. This idea comes from lib2to3. The prefix contains all information
    that is irrelevant for the parser like newlines in parentheses or comments.
    """

    def dedent_if_necessary(start):
        while start < indents[-1]:
            if start > indents[-2]:
                yield PythonToken(ERROR_DEDENT, '', (lnum, start), '')
                indents[-1] = start
                break
            indents.pop()
            yield PythonToken(DEDENT, '', spos, '')
    pseudo_token, single_quoted, triple_quoted, endpats, whitespace, fstring_pattern_map, always_break_tokens = _get_token_collection(version_info)
    paren_level = 0
    if indents is None:
        indents = [0]
    max_ = 0
    numchars = '0123456789'
    contstr = ''
    contline: str
    contstr_start: Tuple[int, int]
    endprog: Pattern
    new_line = True
    prefix = ''
    additional_prefix = ''
    lnum = start_pos[0] - 1
    fstring_stack: List[FStringNode] = []
    for line in lines:
        lnum += 1
        pos = 0
        max_ = len(line)
        if is_first_token:
            if line.startswith(BOM_UTF8_STRING):
                additional_prefix = BOM_UTF8_STRING
                line = line[1:]
                max_ = len(line)
            line = '^' * start_pos[1] + line
            pos = start_pos[1]
            max_ += start_pos[1]
            is_first_token = False
        if contstr:
            endmatch = endprog.match(line)
            if endmatch:
                pos = endmatch.end(0)
                yield PythonToken(STRING, contstr + line[:pos], contstr_start, prefix)
                contstr = ''
                contline = ''
            else:
                contstr = contstr + line
                contline = contline + line
                continue
        while pos < max_:
            if fstring_stack:
                tos = fstring_stack[-1]
                if not tos.is_in_expr():
                    string, pos = _find_fstring_string(endpats, fstring_stack, line, lnum, pos)
                    if string:
                        yield PythonToken(FSTRING_STRING, string, tos.last_string_start_pos, prefix='')
                        tos.previous_lines = ''
                        continue
                    if pos == max_:
                        break
                rest = line[pos:]
                fstring_end_token, additional_prefix, quote_length = _close_fstring_if_necessary(fstring_stack, rest, lnum, pos, additional_prefix)
                pos += quote_length
                if fstring_end_token is not None:
                    yield fstring_end_token
                    continue
            if fstring_stack:
                string_line = line
                for fstring_stack_node in fstring_stack:
                    quote = fstring_stack_node.quote
                    end_match = endpats[quote].match(line, pos)
                    if end_match is not None:
                        end_match_string = end_match.group(0)
                        if len(end_match_string) - len(quote) + pos < len(string_line):
                            string_line = line[:pos] + end_match_string[:-len(quote)]
                pseudomatch = pseudo_token.match(string_line, pos)
            else:
                pseudomatch = pseudo_token.match(line, pos)
            if pseudomatch:
                prefix = additional_prefix + pseudomatch.group(1)
                additional_prefix = ''
                start, pos = pseudomatch.span(2)
                spos = (lnum, start)
                token = pseudomatch.group(2)
                if token == '':
                    assert prefix
                    additional_prefix = prefix
                    break
                initial = token[0]
            else:
                match = whitespace.match(line, pos)
                initial = line[match.end()]
                start = match.end()
                spos = (lnum, start)
            if new_line and initial not in '\r\n#' and (initial != '\\' or pseudomatch is None):
                new_line = False
                if paren_level == 0 and (not fstring_stack):
                    indent_start = start
                    if indent_start > indents[-1]:
                        yield PythonToken(INDENT, '', spos, '')
                        indents.append(indent_start)
                    yield from dedent_if_necessary(indent_start)
            if not pseudomatch:
                match = whitespace.match(line, pos)
                if new_line and paren_level == 0 and (not fstring_stack):
                    yield from dedent_if_necessary(match.end())
                pos = match.end()
                new_line = False
                yield PythonToken(ERRORTOKEN, line[pos], (lnum, pos), additional_prefix + match.group(0))
                additional_prefix = ''
                pos += 1
                continue
            if initial in numchars or (initial == '.' and token != '.' and (token != '...')):
                yield PythonToken(NUMBER, token, spos, prefix)
            elif pseudomatch.group(3) is not None:
                if token in always_break_tokens and (fstring_stack or paren_level):
                    fstring_stack[:] = []
                    paren_level = 0
                    m = re.match('[ \\f\\t]*$', line[:start])
                    if m is not None:
                        yield from dedent_if_necessary(m.end())
                if token.isidentifier():
                    yield PythonToken(NAME, token, spos, prefix)
                else:
                    yield from _split_illegal_unicode_name(token, spos, prefix)
            elif initial in '\r\n':
                if any((not f.allow_multiline() for f in fstring_stack)):
                    fstring_stack.clear()
                if not new_line and paren_level == 0 and (not fstring_stack):
                    yield PythonToken(NEWLINE, token, spos, prefix)
                else:
                    additional_prefix = prefix + token
                new_line = True
            elif initial == '#':
                assert not token.endswith('\n') and (not token.endswith('\r'))
                if fstring_stack and fstring_stack[-1].is_in_expr():
                    yield PythonToken(ERRORTOKEN, initial, spos, prefix)
                    pos = start + 1
                else:
                    additional_prefix = prefix + token
            elif token in triple_quoted:
                endprog = endpats[token]
                endmatch = endprog.match(line, pos)
                if endmatch:
                    pos = endmatch.end(0)
                    token = line[start:pos]
                    yield PythonToken(STRING, token, spos, prefix)
                else:
                    contstr_start = spos
                    contstr = line[start:]
                    contline = line
                    break
            elif initial in single_quoted or token[:2] in single_quoted or token[:3] in single_quoted:
                if token[-1] in '\r\n':
                    contstr_start = (lnum, start)
                    endprog = endpats.get(initial) or endpats.get(token[1]) or endpats.get(token[2])
                    contstr = line[start:]
                    contline = line
                    break
                else:
                    yield PythonToken(STRING, token, spos, prefix)
            elif token in fstring_pattern_map:
                fstring_stack.append(FStringNode(fstring_pattern_map[token]))
                yield PythonToken(FSTRING_START, token, spos, prefix)
            elif initial == '\\' and line[start:] in ('\\\n', '\\\r\n', '\\\r'):
                additional_prefix += prefix + line[start:]
                break
            else:
                if token in '([{':
                    if fstring_stack:
                        fstring_stack[-1].open_parentheses(token)
                    else:
                        paren_level += 1
                elif token in ')]}':
                    if fstring_stack:
                        fstring_stack[-1].close_parentheses(token)
                    elif paren_level:
                        paren_level -= 1
                elif token.startswith(':') and fstring_stack and (fstring_stack[-1].parentheses_count - fstring_stack[-1].format_spec_count == 1):
                    fstring_stack[-1].format_spec_count += 1
                    token = ':'
                    pos = start + 1
                yield PythonToken(OP, token, spos, prefix)
    if contstr:
        yield PythonToken(ERRORTOKEN, contstr, contstr_start, prefix)
        if contstr.endswith('\n') or contstr.endswith('\r'):
            new_line = True
    if fstring_stack:
        tos = fstring_stack[-1]
        if tos.previous_lines:
            yield PythonToken(FSTRING_STRING, tos.previous_lines, tos.last_string_start_pos, prefix='')
    end_pos = (lnum, max_)
    for indent in indents[1:]:
        indents.pop()
        yield PythonToken(DEDENT, '', end_pos, '')
    yield PythonToken(ENDMARKER, '', end_pos, additional_prefix)