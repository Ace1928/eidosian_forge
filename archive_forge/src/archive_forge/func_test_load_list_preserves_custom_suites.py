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
def test_load_list_preserves_custom_suites(self):
    if testresources is None:
        self.skipTest('Need testresources')
    self.useFixture(SampleResourcedFixture())
    tempdir = self.useFixture(fixtures.TempDir())
    tempname = tempdir.path + '/tests.list'
    f = open(tempname, 'wb')
    try:
        f.write(_b('\ntesttools.resourceexample.TestFoo.test_bar\ntesttools.resourceexample.TestFoo.test_foo\n'))
    finally:
        f.close()
    stdout = self.useFixture(fixtures.StringStream('stdout'))
    with fixtures.MonkeyPatch('sys.stdout', stdout.stream):
        try:
            run.main(['prog', '--load-list', tempname, 'testtools.resourceexample.test_suite'], stdout.stream)
        except SystemExit:
            pass
    out = stdout.getDetails()['stdout'].as_text()
    self.assertEqual(1, out.count('Setting up Printer'), '%r' % out)