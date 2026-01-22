import sys
import unittest
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.base import Response, LazyObject
from libcloud.common.exceptions import BaseHTTPError, RateLimitReachedError
def test_lazy_init(self):
    a = self.A(1, y=2)
    self.assertTrue(isinstance(a, self.A))
    with mock.patch.object(self.A, '__init__', return_value=None) as mock_init:
        a = self.A.lazy(3, y=4)
        self.assertTrue(isinstance(a, self.A))
        mock_init.assert_not_called()
        self.assertEqual(a.__dict__, {})
        mock_init.assert_called_once_with(3, y=4)