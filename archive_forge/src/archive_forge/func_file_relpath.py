import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def file_relpath(base: str, path: str) -> str:
    """Compute just the relative sub-portion of a url

    This assumes that both paths are already fully specified file:// URLs.
    """
    if len(base) < MIN_ABS_FILEURL_LENGTH:
        raise ValueError('Length of base (%r) must equal or exceed the platform minimum url length (which is %d)' % (base, MIN_ABS_FILEURL_LENGTH))
    base = osutils.normpath(local_path_from_url(base))
    path = osutils.normpath(local_path_from_url(path))
    return escape(osutils.relpath(base, path))