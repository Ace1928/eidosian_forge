from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
@property
def request_uri(self):
    """Absolute path including the query string."""
    uri = self.path or '/'
    if self.query is not None:
        uri += '?' + self.query
    return uri