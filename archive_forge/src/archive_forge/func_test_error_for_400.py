import unittest
from lazr.restfulclient.errors import (
def test_error_for_400(self):
    """Make sure a 400 response code yields ResponseError."""
    self.error_for_status(400, ResponseError, 'error message')