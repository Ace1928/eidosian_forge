import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def guess_lexer_for_filename(_fn, _text, **options):
    """
    Lookup all lexers that handle those filenames primary (``filenames``)
    or secondary (``alias_filenames``). Then run a text analysis for those
    lexers and choose the best result.

    usage::

        >>> from pygments.lexers import guess_lexer_for_filename
        >>> guess_lexer_for_filename('hello.html', '<%= @foo %>')
        <pygments.lexers.templates.RhtmlLexer object at 0xb7d2f32c>
        >>> guess_lexer_for_filename('hello.html', '<h1>{{ title|e }}</h1>')
        <pygments.lexers.templates.HtmlDjangoLexer object at 0xb7d2f2ac>
        >>> guess_lexer_for_filename('style.css', 'a { color: <?= $link ?> }')
        <pygments.lexers.templates.CssPhpLexer object at 0xb7ba518c>
    """
    fn = basename(_fn)
    primary = {}
    matching_lexers = set()
    for lexer in _iter_lexerclasses():
        for filename in lexer.filenames:
            if _fn_matches(fn, filename):
                matching_lexers.add(lexer)
                primary[lexer] = True
        for filename in lexer.alias_filenames:
            if _fn_matches(fn, filename):
                matching_lexers.add(lexer)
                primary[lexer] = False
    if not matching_lexers:
        raise ClassNotFound('no lexer for filename %r found' % fn)
    if len(matching_lexers) == 1:
        return matching_lexers.pop()(**options)
    result = []
    for lexer in matching_lexers:
        rv = lexer.analyse_text(_text)
        if rv == 1.0:
            return lexer(**options)
        result.append((rv, lexer))

    def type_sort(t):
        return (t[0], primary[t[1]], t[1].priority, t[1].__name__)
    result.sort(key=type_sort)
    return result[-1][1](**options)