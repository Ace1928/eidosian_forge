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
def got_timestamp(self, timestamp):
    """Called when we receive a timestamp.

        This will always update the second element of the 'timestamps' tuple.
        It doesn't compare timestamps at all.
        """
    return self.set(timestamps=(self.timestamps[0], timestamp))