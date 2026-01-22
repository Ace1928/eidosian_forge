import logging
import unittest
import mock
import numpy as np
from gensim.parsing.preprocessing import (
def test_strip_tags(self):
    self.assertEqual(strip_tags('<i>Hello</i> <b>World</b>!'), 'Hello World!')