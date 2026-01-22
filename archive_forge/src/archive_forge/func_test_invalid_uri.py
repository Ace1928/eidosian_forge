from collections import defaultdict
import unittest
from lazr.uri import (
def test_invalid_uri(self):
    self.assertRaises(InvalidURIError, URI, 'http://€xample.com/')