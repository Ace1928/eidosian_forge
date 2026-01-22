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
def test_stdout_honoured(self):
    self.useFixture(SampleTestFixture())
    tests = []
    out = io.StringIO()
    exc = self.assertRaises(SystemExit, run.main, argv=['prog', 'testtools.runexample.test_suite'], stdout=out)
    self.assertEqual((0,), exc.args)
    self.assertThat(out.getvalue(), MatchesRegex('Tests running...\n\nRan 2 tests in \\d.\\d\\d\\ds\nOK\n'))