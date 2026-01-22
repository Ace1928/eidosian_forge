from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importfrom_relative(self):
    binding = ImportationFrom('a', None, '.', 'a')
    assert binding.source_statement == 'from . import a'
    assert str(binding) == '.a'