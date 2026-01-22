from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_star(self):
    binding = StarImportation('a.b', None)
    assert binding.source_statement == 'from a.b import *'
    assert str(binding) == 'a.b.*'