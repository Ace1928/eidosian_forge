import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def split_segment_parameters_raw(url):
    """Split the subsegment of the last segment of a URL.

    Args:
      url: A relative or absolute URL
    Returns: (url, subsegments)
    """
    lurl = strip_trailing_slash(url)
    segment_start = lurl.find(',', lurl.rfind('/') + 1)
    if segment_start == -1:
        return (url, [])
    return (lurl[:segment_start], [str(s) for s in lurl[segment_start + 1:].split(',')])