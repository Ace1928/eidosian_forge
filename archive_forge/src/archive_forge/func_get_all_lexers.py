import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def get_all_lexers():
    """Return a generator of tuples in the form ``(name, aliases,
    filenames, mimetypes)`` of all know lexers.
    """
    for item in itervalues(LEXERS):
        yield item[1:]
    for lexer in find_plugin_lexers():
        yield (lexer.name, lexer.aliases, lexer.filenames, lexer.mimetypes)