import doctest
import io
import sys
from textwrap import dedent
import unittest
from unittest import TestSuite
import testtools
from testtools import TestCase, run, skipUnless
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools import TestCase
from fixtures import Fixture
from testresources import (
from testtools import TestCase
from testtools import TestCase, clone_test_with_new_id
def test_run_list(self):
    self.useFixture(SampleTestFixture())
    out = io.StringIO()
    try:
        run.main(['prog', '-l', 'testtools.runexample.test_suite'], out)
    except SystemExit:
        exc_info = sys.exc_info()
        raise AssertionError('-l tried to exit. %r' % exc_info[1])
    self.assertEqual('testtools.runexample.TestFoo.test_bar\ntesttools.runexample.TestFoo.test_quux\n', out.getvalue())