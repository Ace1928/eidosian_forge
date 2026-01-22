from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_ok_percent_format_cannot_determine_element_count(self):
    self.flakes("\n            a = []\n            '%s %s' % [*a]\n            '%s %s' % (*a,)\n        ")
    self.flakes("\n            k = {}\n            '%(k)s' % {**k}\n        ")