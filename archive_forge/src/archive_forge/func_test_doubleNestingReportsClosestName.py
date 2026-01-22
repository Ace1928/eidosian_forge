import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_doubleNestingReportsClosestName(self):
    """
        Test that referencing a local name in a nested scope that shadows a
        variable declared in two different outer scopes before it is defined
        in the innermost scope generates an UnboundLocal warning which
        refers to the nearest shadowed name.
        """
    exc = self.flakes('\n            def a():\n                x = 1\n                def b():\n                    x = 2 # line 5\n                    def c():\n                        x\n                        x = 3\n                        return x\n                    return x\n                return x\n        ', m.UndefinedLocal).messages[0]
    expected_line_num = 7 if self.withDoctest else 5
    self.assertEqual(exc.message_args, ('x', expected_line_num))