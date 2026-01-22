from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def replacer(match):
    return needles_and_replacements[match.group(0)]