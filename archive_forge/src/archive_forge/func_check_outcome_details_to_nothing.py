import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def check_outcome_details_to_nothing(self, outcome, expected=None):
    """Call an outcome with a details dict to be swallowed."""
    if not expected:
        expected = outcome
    details = {'foo': 'bar'}
    getattr(self.converter, outcome)(self, details=details)
    self.assertEqual([(expected, self)], self.result._events)