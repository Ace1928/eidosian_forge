import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_QueryDict(self):
    qd = QueryDict()
    self.assertTrue(isinstance(qd, dict))
    qd.toto = 1
    self.assertEqual(qd.get('toto'), 1)
    toto = qd.toto
    self.assertEqual(toto, 1)
    with self.assertRaises(AttributeError):
        foo = qd.not_an_attribute