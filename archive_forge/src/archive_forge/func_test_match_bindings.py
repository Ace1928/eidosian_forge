from sys import version_info
from pyflakes.test.harness import TestCase, skipIf
def test_match_bindings(self):
    self.flakes("\n            def f():\n                x = 1\n                match x:\n                    case 1 as y:\n                        print(f'matched as {y}')\n        ")
    self.flakes("\n            def f():\n                x = [1, 2, 3]\n                match x:\n                    case [1, y, 3]:\n                        print(f'matched {y}')\n        ")
    self.flakes("\n            def f():\n                x = {'foo': 1}\n                match x:\n                    case {'foo': y}:\n                        print(f'matched {y}')\n        ")