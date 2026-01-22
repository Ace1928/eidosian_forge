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
def onException(self, exc_info, tb_label='traceback'):
    """Called when an exception propagates from test code.

        :seealso addOnException:
        """
    if exc_info[0] not in [self.skipException, _UnexpectedSuccess, _ExpectedFailure]:
        self._report_traceback(exc_info, tb_label=tb_label)
    for handler in self.__exception_handlers:
        handler(exc_info)