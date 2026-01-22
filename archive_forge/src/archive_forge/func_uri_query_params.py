from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
@property
def uri_query_params(self):
    if not self.uri_query:
        return []
    return urlparse.parse_qsl(self.uri_query, keep_blank_values=True, strict_parsing=True)