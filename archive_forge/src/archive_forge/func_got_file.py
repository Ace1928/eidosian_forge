import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def got_file(self, file_name, file_bytes, mime_type=None):
    """Called when we receive file information.

        ``mime_type`` is only used when this is the first time we've seen data
        from this file.
        """
    if file_name in self.details:
        case = self
    else:
        content_type = _make_content_type(mime_type)
        content_bytes = []
        case = self.transform(['details', file_name], Content(content_type, lambda: content_bytes))
    case.details[file_name].iter_bytes().append(file_bytes)
    return case