from __future__ import annotations
import warnings
from . import assertions
from .. import exc
from .. import exc as sa_exc
from ..exc import SATestSuiteWarning
from ..util.langhelpers import _warnings_warn
def assert_warnings(fn, warning_msgs, regex=False):
    """Assert that each of the given warnings are emitted by fn.

    Deprecated.  Please use assertions.expect_warnings().

    """
    with assertions._expect_warnings(sa_exc.SAWarning, warning_msgs, regex=regex):
        return fn()