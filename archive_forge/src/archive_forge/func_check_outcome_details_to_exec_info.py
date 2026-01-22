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
def check_outcome_details_to_exec_info(self, outcome, expected=None):
    """Call an outcome with a details dict to be made into exc_info."""
    if not expected:
        expected = outcome
    details, err_str = self.get_details_and_string()
    getattr(self.converter, outcome)(self, details=details)
    err = self.converter._details_to_exc_info(details)
    self.assertEqual([(expected, self, err)], self.result._events)