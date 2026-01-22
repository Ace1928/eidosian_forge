import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
def expectThat(self, matchee, matcher, message='', verbose=False):
    """Check that matchee is matched by matcher, but delay the assertion failure.

        This method behaves similarly to ``assertThat``, except that a failed
        match does not exit the test immediately. The rest of the test code
        will continue to run, and the test will be marked as failing after the
        test has finished.

        :param matchee: An object to match with matcher.
        :param matcher: An object meeting the testtools.Matcher protocol.
        :param message: If specified, show this message with any failed match.

        """
    mismatch_error = self._matchHelper(matchee, matcher, message, verbose)
    if mismatch_error is not None:
        self.addDetailUniqueName('Failed expectation', content.StacktraceContent(postfix_content='MismatchError: ' + str(mismatch_error)))
        self.force_failure = True