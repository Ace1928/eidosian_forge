import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def get_lexer_for_filename(_fn, code=None, **options):
    """Get a lexer for a filename.

    If multiple lexers match the filename pattern, use ``analyse_text()`` to
    figure out which one is more appropriate.

    Raises ClassNotFound if not found.
    """
    res = find_lexer_class_for_filename(_fn, code)
    if not res:
        raise ClassNotFound('no lexer for filename %r found' % _fn)
    return res(**options)