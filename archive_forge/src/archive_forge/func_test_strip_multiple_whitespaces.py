import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_strip_multiple_whitespaces(self):
    self.assertEqual(strip_multiple_whitespaces('salut  les\r\nloulous!'), 'salut les loulous!')