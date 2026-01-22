import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_strip_numeric(self):
    self.assertEqual(strip_numeric('salut les amis du 59'), 'salut les amis du ')