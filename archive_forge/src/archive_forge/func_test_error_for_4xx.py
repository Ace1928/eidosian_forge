import unittest
from lazr.restfulclient.errors import (
def test_error_for_4xx(self):
    """Make sure an unrexognized 4xx response code yields ClientError."""
    self.error_for_status(499, ClientError, 'error message')