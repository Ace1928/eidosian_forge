import sys
from testtools.testresult import ExtendedToOriginalDecorator
Called when user code raises an exception.

        If 'exc_info' is a `MultipleExceptions`, then we recurse into it
        unpacking the errors that it's made up from.

        :param exc_info: A sys.exc_info() tuple for the user error.
        :param tb_label: An optional string label for the error.  If
            not specified, will default to 'traceback'.
        :return: 'exception_caught' if we catch one of the exceptions that
            have handlers in 'handlers', otherwise raise the error.
        