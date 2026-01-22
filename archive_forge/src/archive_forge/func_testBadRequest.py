import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testBadRequest(self):
    err = exceptions.HttpError.FromResponse(_MakeResponse(400))
    self.assertIsInstance(err, exceptions.HttpError)
    self.assertIsInstance(err, exceptions.HttpBadRequestError)
    self.assertEquals(err.status_code, 400)