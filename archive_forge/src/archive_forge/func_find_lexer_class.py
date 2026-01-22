import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def find_lexer_class(name):
    """Lookup a lexer class by name.

    Return None if not found.
    """
    if name in _lexer_cache:
        return _lexer_cache[name]
    for module_name, lname, aliases, _, _ in itervalues(LEXERS):
        if name == lname:
            _load_lexers(module_name)
            return _lexer_cache[name]
    for cls in find_plugin_lexers():
        if cls.name == name:
            return cls