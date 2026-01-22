from unittest import TestCase
from fastimport import (
def test_MissingSection(self):
    e = errors.MissingSection(99, 'foo', 'bar')
    self.assertEqual('line 99: Command foo is missing section bar', str(e))