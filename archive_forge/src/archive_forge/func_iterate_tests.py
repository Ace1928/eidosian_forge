from collections import Counter
from pprint import pformat
from queue import Queue
import sys
import threading
import unittest
import testtools
def iterate_tests(test_suite_or_case):
    """Iterate through all of the test cases in 'test_suite_or_case'."""
    try:
        suite = iter(test_suite_or_case)
    except TypeError:
        yield test_suite_or_case
    else:
        for test in suite:
            yield from iterate_tests(test)