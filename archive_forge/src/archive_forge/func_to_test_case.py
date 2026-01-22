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
def to_test_case(self):
    """Convert into a TestCase object.

        :return: A PlaceHolder test object.
        """
    global PlaceHolder
    if PlaceHolder is None:
        from testtools.testcase import PlaceHolder
    outcome = _status_map[self.status]
    return PlaceHolder(self.id, outcome=outcome, details=self.details, tags=self.tags, timestamps=self.timestamps)