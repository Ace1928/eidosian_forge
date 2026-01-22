import re
import sys
def shebang_matches(text, regex):
    """Check if the given regular expression matches the last part of the
    shebang if one exists.

        >>> from pygments.util import shebang_matches
        >>> shebang_matches('#!/usr/bin/env python', r'python(2\\.\\d)?')
        True
        >>> shebang_matches('#!/usr/bin/python2.4', r'python(2\\.\\d)?')
        True
        >>> shebang_matches('#!/usr/bin/python-ruby', r'python(2\\.\\d)?')
        False
        >>> shebang_matches('#!/usr/bin/python/ruby', r'python(2\\.\\d)?')
        False
        >>> shebang_matches('#!/usr/bin/startsomethingwith python',
        ...                 r'python(2\\.\\d)?')
        True

    It also checks for common windows executable file extensions::

        >>> shebang_matches('#!C:\\\\Python2.4\\\\Python.exe', r'python(2\\.\\d)?')
        True

    Parameters (``'-f'`` or ``'--foo'`` are ignored so ``'perl'`` does
    the same as ``'perl -e'``)

    Note that this method automatically searches the whole string (eg:
    the regular expression is wrapped in ``'^$'``)
    """
    index = text.find('\n')
    if index >= 0:
        first_line = text[:index].lower()
    else:
        first_line = text.lower()
    if first_line.startswith('#!'):
        try:
            found = [x for x in split_path_re.split(first_line[2:].strip()) if x and (not x.startswith('-'))][-1]
        except IndexError:
            return False
        regex = re.compile('^%s(\\.(exe|cmd|bat|bin))?$' % regex, re.IGNORECASE)
        if regex.search(found) is not None:
            return True
    return False