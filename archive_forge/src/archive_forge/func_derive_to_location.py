import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def derive_to_location(from_location):
    """Derive a TO_LOCATION given a FROM_LOCATION.

    The normal case is a FROM_LOCATION of http://foo/bar => bar.
    The Right Thing for some logical destinations may differ though
    because no / may be present at all. In that case, the result is
    the full name without the scheme indicator, e.g. lp:foo-bar => foo-bar.
    This latter case also applies when a Windows drive
    is used without a path, e.g. c:foo-bar => foo-bar.
    If no /, path separator or : is found, the from_location is returned.
    """
    from_location = strip_segment_parameters(from_location)
    if from_location.find('/') >= 0 or from_location.find(os.sep) >= 0:
        return os.path.basename(from_location.rstrip('/\\'))
    else:
        sep = from_location.find(':')
        if sep > 0:
            return from_location[sep + 1:]
        else:
            return from_location