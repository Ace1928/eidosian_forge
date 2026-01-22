import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testUnknownStatus(self):
    err = exceptions.HttpError.FromResponse(_MakeResponse(499))
    self.assertIsInstance(err, exceptions.HttpError)
    self.assertEquals(err.status_code, 499)