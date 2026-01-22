from __future__ import unicode_literals
import codecs
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import scan_bibtex_string
from pybtex.database.output import BaseWriter
def check_braces(self, s):
    """
        Raise an exception if the given string has unmatched braces.

        >>> w = Writer()
        >>> w.check_braces('Cat eats carrots.')
        >>> w.check_braces('Cat eats {carrots}.')
        >>> w.check_braces('Cat eats {carrots{}}.')
        >>> w.check_braces('')
        >>> w.check_braces('end}')
        >>> try:
        ...     w.check_braces('{')
        ... except BibTeXError as error:
        ...     print(error)
        String has unmatched braces: {
        >>> w.check_braces('{test}}')
        >>> try:
        ...     w.check_braces('{{test}')
        ... except BibTeXError as error:
        ...     print(error)
        String has unmatched braces: {{test}

        """
    tokens = list(scan_bibtex_string(s))
    if tokens:
        end_brace_level = tokens[-1][1]
        if end_brace_level != 0:
            raise BibTeXError('String has unmatched braces: %s' % s)