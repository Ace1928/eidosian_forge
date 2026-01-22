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
def test_assertRaises_function_repr_in_exception(self):

    def foo():
        """An arbitrary function."""
        pass
    self.assertThat(lambda: self.assertRaises(Exception, foo), Raises(MatchesException(self.failureException, f'.*{foo!r}.*')))