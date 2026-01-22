import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testMalformedStatus(self):
    err = exceptions.HttpError.FromResponse(_MakeResponse('BAD'))
    self.assertIsInstance(err, exceptions.HttpError)