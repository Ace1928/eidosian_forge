import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testDefEnum(self):
    """Test def_enum works by building enum class from dict."""
    WeekDay = messages.Enum.def_enum({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 6, 'Saturday': 7, 'Sunday': 8}, 'WeekDay')
    self.assertEquals('Wednesday', WeekDay(3).name)
    self.assertEquals(6, WeekDay('Friday').number)
    self.assertEquals(WeekDay.Sunday, WeekDay('Sunday'))