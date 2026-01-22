import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testConflict(self):
    err = exceptions.HttpError.FromResponse(_MakeResponse(409))
    self.assertIsInstance(err, exceptions.HttpError)
    self.assertIsInstance(err, exceptions.HttpConflictError)
    self.assertEquals(err.status_code, 409)