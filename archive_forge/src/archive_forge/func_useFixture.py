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
def useFixture(self, fixture):
    """Use fixture in a test case.

        The fixture will be setUp, and self.addCleanup(fixture.cleanUp) called.

        :param fixture: The fixture to use.
        :return: The fixture, after setting it up and scheduling a cleanup for
           it.
        """
    try:
        fixture.setUp()
    except MultipleExceptions as e:
        if fixtures is not None and e.args[-1][0] is fixtures.fixture.SetupError:
            gather_details(e.args[-1][1].args[0], self.getDetails())
        raise
    except:
        exc_info = sys.exc_info()
        try:
            if hasattr(fixture, '_details') and fixture._details is not None:
                gather_details(fixture.getDetails(), self.getDetails())
        except:
            self._report_traceback(exc_info)
            raise
        else:
            reraise(*exc_info)
    else:
        self.addCleanup(fixture.cleanUp)
        self.addCleanup(gather_details, fixture.getDetails(), self.getDetails())
        return fixture