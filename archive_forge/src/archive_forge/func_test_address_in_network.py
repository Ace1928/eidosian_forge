import os
import unittest
from websocket._url import (
def test_address_in_network(self):
    self.assertTrue(_is_address_in_network('127.0.0.1', '127.0.0.0/8'))
    self.assertTrue(_is_address_in_network('127.1.0.1', '127.0.0.0/8'))
    self.assertFalse(_is_address_in_network('127.1.0.1', '127.0.0.0/24'))