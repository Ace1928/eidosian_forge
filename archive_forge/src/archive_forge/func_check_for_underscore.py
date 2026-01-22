import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
def check_for_underscore(self):
    """Check if the user has set the '_' variable by hand."""
    if '_' in builtin_mod.__dict__:
        try:
            user_value = self.shell.user_ns['_']
            if user_value is not self._:
                return
            del self.shell.user_ns['_']
        except KeyError:
            pass