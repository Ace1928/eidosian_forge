import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_mod(self):
    """Test 'import root-XXX.mod-XXX as mod2'"""
    try:
        mod1
    except NameError:
        pass
    else:
        self.fail('mod1 was not supposed to exist yet')
    mod_path = self.root_name + '.' + self.mod_name
    InstrumentedImportReplacer(scope=globals(), name='mod1', module_path=[self.root_name, self.mod_name], member=None, children={})
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(mod1, '__class__'))
    self.assertEqual(2, mod1.var2)
    self.assertEqual('y', mod1.func2('y'))
    self.assertEqual([('__getattribute__', 'var2'), ('_import', 'mod1'), ('import', mod_path, [], 0)], self.actions)