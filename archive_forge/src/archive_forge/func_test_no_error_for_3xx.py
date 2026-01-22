import unittest
from lazr.restfulclient.errors import (
def test_no_error_for_3xx(self):
    """Make sure a 3xx response code yields no error."""
    for status in (301, 302, 303, 304, 399):
        self.error_for_status(status, None)