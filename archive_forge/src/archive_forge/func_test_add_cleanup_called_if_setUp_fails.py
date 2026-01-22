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
def test_add_cleanup_called_if_setUp_fails(self):
    log = []

    def broken_set_up(ignored):
        log.append('brokenSetUp')
        raise RuntimeError('Deliberate broken setUp')
    test = make_test_case(self.getUniqueString(), set_up=broken_set_up, test_body=lambda _: log.append('runTest'), tear_down=lambda _: log.append('tearDown'), cleanups=[lambda _: log.append('cleanup')])
    test.run()
    self.assertThat(log, Equals(['brokenSetUp', 'cleanup']))