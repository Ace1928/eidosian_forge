import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log
def exclude_pattern(self, pattern, anchor=1, prefix=None, is_regex=0):
    """Remove strings (presumably filenames) from 'files' that match
        'pattern'.  Other parameters are the same as for
        'include_pattern()', above.
        The list 'self.files' is modified in place.
        Return True if files are found, False otherwise.
        """
    files_found = False
    pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
    self.debug_print("exclude_pattern: applying regex r'%s'" % pattern_re.pattern)
    for i in range(len(self.files) - 1, -1, -1):
        if pattern_re.search(self.files[i]):
            self.debug_print(' removing ' + self.files[i])
            del self.files[i]
            files_found = True
    return files_found