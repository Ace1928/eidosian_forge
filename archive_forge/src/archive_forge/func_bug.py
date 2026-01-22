import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@property
def bug(self):
    """ list of bug numbers that had requested the package removal

        The bug numbers are returned as integers.

        Note: there is normally only one entry in this list but there may be
        more than one.
        """
    if 'bug' not in self:
        return []
    return [int(b) for b in self['bug'].split(',')]