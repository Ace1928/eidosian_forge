import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
def set_up_done(exception_caught):
    """Set up is done, either clean up or run the test."""
    if self.exception_caught == exception_caught:
        fails.append(None)
        return clean_up()
    else:
        d = self._run_user(self.case._run_test_method, self.result)
        d.addCallback(fail_if_exception_caught)
        d.addBoth(tear_down)
        return d