import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def set_end_time(self, timestamp_in_secs):
    """Sets the start timestamp of this test suite.

    Args:
      timestamp_in_secs: timestamp in seconds since epoch
    """
    self.overall_end_time = timestamp_in_secs