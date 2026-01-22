from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
@property
def uri_query(self):
    return urlparse.urlparse(self.uri).query