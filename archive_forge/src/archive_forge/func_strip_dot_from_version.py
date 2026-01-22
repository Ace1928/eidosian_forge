import logging
import re
import statsd
import webob.dec
from oslo_middleware import base
@staticmethod
def strip_dot_from_version(path):
    match = VERSION_REGEX.match(path)
    if match is None:
        return path
    return path.replace(match.group(1), match.group(1).replace('.', ''))