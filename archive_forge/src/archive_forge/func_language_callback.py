import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
def language_callback(lexer, match):
    """Parse the content of a $-string using a lexer

    The lexer is chosen looking for a nearby LANGUAGE or assumed as
    plpgsql if inside a DO statement and no LANGUAGE has been found.
    """
    l = None
    m = language_re.match(lexer.text[match.end():match.end() + 100])
    if m is not None:
        l = lexer._get_lexer(m.group(1))
    else:
        m = list(language_re.finditer(lexer.text[max(0, match.start() - 100):match.start()]))
        if m:
            l = lexer._get_lexer(m[-1].group(1))
        else:
            m = list(do_re.finditer(lexer.text[max(0, match.start() - 25):match.start()]))
            if m:
                l = lexer._get_lexer('plpgsql')
    yield (match.start(1), String, match.group(1))
    yield (match.start(2), String.Delimiter, match.group(2))
    yield (match.start(3), String, match.group(3))
    if l:
        for x in l.get_tokens_unprocessed(match.group(4)):
            yield x
    else:
        yield (match.start(4), String, match.group(4))
    yield (match.start(5), String, match.group(5))
    yield (match.start(6), String.Delimiter, match.group(6))
    yield (match.start(7), String, match.group(7))