import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testExceptionMessageIncludesErrorDetails(self):
    err = exceptions.HttpError.FromResponse(_MakeResponse(403))
    self.assertIn('403', repr(err))
    self.assertIn('http://www.google.com', repr(err))
    self.assertIn('{"field": "abc"}', repr(err))