from testtools.matchers import Mismatch
from ._deferred import failure_content, on_deferred_result
def has_no_result():
    """Match a Deferred that has not yet fired.

    For example, this will pass::

        assert_that(defer.Deferred(), has_no_result())

    But this will fail:

    >>> assert_that(defer.succeed(None), has_no_result())
    Traceback (most recent call last):
      ...
      File "testtools/assertions.py", line 22, in assert_that
        raise MismatchError(matchee, matcher, mismatch, verbose)
    testtools.matchers._impl.MismatchError: No result expected on <Deferred at ... current result: None>, found None instead

    As will this:

    >>> assert_that(defer.fail(RuntimeError('foo')), has_no_result())
    Traceback (most recent call last):
      ...
      File "testtools/assertions.py", line 22, in assert_that
        raise MismatchError(matchee, matcher, mismatch, verbose)
    testtools.matchers._impl.MismatchError: No result expected on <Deferred at ... current result: <twisted.python.failure.Failure <type 'exceptions.RuntimeError'>>>, found <twisted.python.failure.Failure <type 'exceptions.RuntimeError'>> instead
    """
    return _NO_RESULT