from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
@fix_unicode_literals_in_doctest
def split_tex_string(string, sep=None, strip=True, filter_empty=False):
    """Split a string using the given separator (regexp).

    Everything at brace level > 0 is ignored.

    >>> split_tex_string('')
    []
    >>> split_tex_string('     ')
    []
    >>> split_tex_string('.a.b.c.', r'\\.')
    [u'', u'a', u'b', u'c', u'']
    >>> split_tex_string('.a.b.c.{d.}.', r'\\.')
    [u'', u'a', u'b', u'c', u'{d.}', u'']
    >>> split_tex_string('Matsui      Fuuka')
    [u'Matsui', u'Fuuka']
    >>> split_tex_string('{Matsui      Fuuka}')
    [u'{Matsui      Fuuka}']
    >>> split_tex_string(r'Matsui\\ Fuuka')
    [u'Matsui', u'Fuuka']
    >>> split_tex_string(r'{Matsui\\ Fuuka}')
    [u'{Matsui\\\\ Fuuka}']
    >>> split_tex_string('a')
    [u'a']
    >>> split_tex_string('on a')
    [u'on', u'a']
    >>> split_tex_string(r'Qui\\~{n}onero-Candela, J.')
    [u'Qui\\\\~{n}onero-Candela,', u'J.']
    """
    if sep is None:
        sep = BIBTEX_SPACE_RE
        filter_empty = True
    sep = re.compile(sep)
    result = []
    word_parts = []
    while True:
        head, brace, string = string.partition('{')
        if head:
            head_parts = sep.split(head)
            for word in head_parts[:-1]:
                result.append(''.join(word_parts + [word]))
                word_parts = []
            word_parts.append(head_parts[-1])
        if brace:
            word_parts.append(brace)
            up_to_closing_brace, string = _find_closing_brace(string)
            word_parts.append(up_to_closing_brace)
        else:
            break
    if word_parts:
        result.append(''.join(word_parts))
    if strip:
        result = [part.strip() for part in result]
    if filter_empty:
        result = [part for part in result if part]
    return result