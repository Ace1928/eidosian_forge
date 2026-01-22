from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def makePlaceHolder(self, test_id='foo', error=None, short_description=None):
    if error is None:
        error = self.makeException()
    return ErrorHolder(test_id, error, short_description)