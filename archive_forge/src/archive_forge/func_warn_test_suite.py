from __future__ import annotations
import warnings
from . import assertions
from .. import exc
from .. import exc as sa_exc
from ..exc import SATestSuiteWarning
from ..util.langhelpers import _warnings_warn
def warn_test_suite(message):
    _warnings_warn(message, category=SATestSuiteWarning)