import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def rebase_url(url, old_base, new_base):
    """Convert a relative path from an old base URL to a new base URL.

    The result will be a relative path.
    Absolute paths and full URLs are returned unaltered.
    """
    scheme, separator = _find_scheme_and_separator(url)
    if scheme is not None:
        return url
    if _is_absolute(url):
        return url
    old_parsed = urlparse.urlparse(old_base)
    new_parsed = urlparse.urlparse(new_base)
    if old_parsed[:2] != new_parsed[:2]:
        raise InvalidRebaseURLs(old_base, new_base)
    return determine_relative_path(new_parsed[2], join(old_parsed[2], url))