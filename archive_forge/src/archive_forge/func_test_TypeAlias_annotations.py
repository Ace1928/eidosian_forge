from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_TypeAlias_annotations(self):
    self.flakes('\n        from typing_extensions import TypeAlias\n        from foo import Bar\n\n        bar: TypeAlias = Bar\n        ')
    self.flakes("\n        from typing_extensions import TypeAlias\n        from foo import Bar\n\n        bar: TypeAlias = 'Bar'\n        ")
    self.flakes('\n        from typing_extensions import TypeAlias\n        from foo import Bar\n\n        class A:\n            bar: TypeAlias = Bar\n        ')
    self.flakes("\n        from typing_extensions import TypeAlias\n        from foo import Bar\n\n        class A:\n            bar: TypeAlias = 'Bar'\n        ")
    self.flakes('\n        from typing_extensions import TypeAlias\n\n        bar: TypeAlias\n        ')
    self.flakes('\n        from typing_extensions import TypeAlias\n        from foo import Bar\n\n        bar: TypeAlias\n        ', m.UnusedImport)