import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_from_class_missing(self):
    """When class has no lineno the old context details are returned"""
    path = 'cls_missing.py'

    class A:
        pass

    class M:
        pass
    context = export_pot._ModuleContext(path, 3, ({'A': 15}, {}))
    contextA = context.from_class(A)
    contextM1 = context.from_class(M)
    self.check_context(contextM1, path, 3)
    contextM2 = contextA.from_class(M)
    self.check_context(contextM2, path, 15)
    self.assertContainsRe(self.get_log(), "Definition of <.*M'> not found")