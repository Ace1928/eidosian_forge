from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def compile_pattern_list(self, patterns):
    """This compiles a pattern-string or a list of pattern-strings.
        Patterns must be a StringType, EOF, TIMEOUT, SRE_Pattern, or a list of
        those. Patterns may also be None which results in an empty list (you
        might do this if waiting for an EOF or TIMEOUT condition without
        expecting any pattern).

        This is used by expect() when calling expect_list(). Thus expect() is
        nothing more than::

             cpl = self.compile_pattern_list(pl)
             return self.expect_list(cpl, timeout)

        If you are using expect() within a loop it may be more
        efficient to compile the patterns first and then call expect_list().
        This avoid calls in a loop to compile_pattern_list()::

             cpl = self.compile_pattern_list(my_pattern)
             while some_condition:
                ...
                i = self.expect_list(cpl, timeout)
                ...
        """
    if patterns is None:
        return []
    if not isinstance(patterns, list):
        patterns = [patterns]
    compile_flags = re.DOTALL
    if self.ignorecase:
        compile_flags = compile_flags | re.IGNORECASE
    compiled_pattern_list = []
    for idx, p in enumerate(patterns):
        if isinstance(p, self.allowed_string_types):
            p = self._coerce_expect_string(p)
            compiled_pattern_list.append(re.compile(p, compile_flags))
        elif p is EOF:
            compiled_pattern_list.append(EOF)
        elif p is TIMEOUT:
            compiled_pattern_list.append(TIMEOUT)
        elif isinstance(p, type(re.compile(''))):
            p = self._coerce_expect_re(p)
            compiled_pattern_list.append(p)
        else:
            self._pattern_type_err(p)
    return compiled_pattern_list