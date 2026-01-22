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
def test_run_locals(self):
    stdout = self.useFixture(fixtures.StringStream('stdout'))

    class Failing(TestCase):

        def test_a(self):
            a = 1
            self.fail('a')
    runner = run.TestToolsTestRunner(tb_locals=True, stdout=stdout.stream)
    runner.run(Failing('test_a'))
    self.assertThat(stdout.getDetails()['stdout'].as_text(), Contains('a = 1'))