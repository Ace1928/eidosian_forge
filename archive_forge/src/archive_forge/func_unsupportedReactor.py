from hamcrest import assert_that, equal_to, has_length
from typing_extensions import NoReturn
from twisted.trial._dist.test.matchers import matches_result
from twisted.trial.reporter import TestResult
from twisted.trial.runner import TestLoader
from twisted.trial.unittest import SynchronousTestCase, TestSuite
from .reactormixins import ReactorBuilder
def unsupportedReactor(self: ReactorBuilder) -> NoReturn:
    """
    A function that can be used as a factory for L{ReactorBuilder} tests but
    which always raises an exception.

    This gives the appearance of a reactor type which is unsupported in the
    current runtime configuration for some reason.
    """
    raise Exception(UNSUPPORTED)