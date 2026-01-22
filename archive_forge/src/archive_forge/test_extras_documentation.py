import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
General test template for error_callback argument.

    :param test: Test case instance.
    :param function: Either try_import or try_imports.
    :param arg: Name or names to import.
    :param expected_error_count: Expected number of calls to the callback.
    :param expect_result: Boolean for whether a module should
        ultimately be returned or not.
    