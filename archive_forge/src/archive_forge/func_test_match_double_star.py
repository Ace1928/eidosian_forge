from sys import version_info
from pyflakes.test.harness import TestCase, skipIf
def test_match_double_star(self):
    self.flakes("\n            x = {'foo': 'bar', 'baz': 'womp'}\n            match x:\n                case {'foo': k1, **rest}:\n                    print(f'{k1=} {rest=}')\n        ")