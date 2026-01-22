import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def raise_syntax_error(self, msg, lineno=None):
    if lineno is None:
        lineno = self.lineno
    raise ParseError(self.name, lineno, msg)