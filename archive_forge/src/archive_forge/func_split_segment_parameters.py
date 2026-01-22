import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def split_segment_parameters(url):
    """Split the segment parameters of the last segment of a URL.

    Args:
      url: A relative or absolute URL
    Returns: (url, segment_parameters)
    """
    base_url, subsegments = split_segment_parameters_raw(url)
    parameters = {}
    for subsegment in subsegments:
        try:
            key, value = subsegment.split('=', 1)
        except ValueError:
            raise InvalidURL(url, 'missing = in subsegment')
        if not isinstance(key, str):
            raise TypeError(key)
        if not isinstance(value, str):
            raise TypeError(value)
        parameters[key] = value
    return (base_url, parameters)