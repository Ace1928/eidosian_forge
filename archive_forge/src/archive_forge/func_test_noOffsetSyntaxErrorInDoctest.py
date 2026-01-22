import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_noOffsetSyntaxErrorInDoctest(self):
    exceptions = self.flakes('\n            def buildurl(base, *args, **kwargs):\n                """\n                >>> buildurl(\'/blah.php\', (\'a\', \'&\'), (\'b\', \'=\')\n                \'/blah.php?a=%26&b=%3D\'\n                >>> buildurl(\'/blah.php\', a=\'&\', \'b\'=\'=\')\n                \'/blah.php?b=%3D&a=%26\'\n                """\n                pass\n            ', m.DoctestSyntaxError, m.DoctestSyntaxError).messages
    exc = exceptions[0]
    self.assertEqual(exc.lineno, 4)
    exc = exceptions[1]
    self.assertEqual(exc.lineno, 6)