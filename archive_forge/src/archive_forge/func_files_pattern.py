import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def files_pattern(self):
    """Returns a regular expression equivalent to the Files globs.

        Caches the result until files is set to a different value.

        Raises ValueError if any of the globs are invalid.
        """
    files_str = self['files']
    if self.__cached_files_pat[0] != files_str:
        self.__cached_files_pat = (files_str, globs_to_re(self.files))
    return self.__cached_files_pat[1]