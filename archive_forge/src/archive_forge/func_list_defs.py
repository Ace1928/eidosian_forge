import json
import os
import re
import shutil
import stat
import tempfile
import types
import weakref
from mako import cache
from mako import codegen
from mako import compat
from mako import exceptions
from mako import runtime
from mako import util
from mako.lexer import Lexer
def list_defs(self):
    """return a list of defs in the template.

        .. versionadded:: 1.0.4

        """
    return [i[7:] for i in dir(self.module) if i[:7] == 'render_']