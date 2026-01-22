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
@skipUnless(fixtures, 'fixtures not present')
def test_issue_16662(self):
    pkg = self.useFixture(SampleLoadTestsPackage())
    out = io.StringIO()
    unittest.defaultTestLoader._top_level_dir = None
    self.assertEqual(None, run.main(['prog', 'discover', '-l', pkg.package.base], out))
    self.assertEqual(dedent('            discoverexample.TestExample.test_foo\n            fred\n            '), out.getvalue())