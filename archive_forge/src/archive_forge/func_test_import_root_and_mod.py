import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_root_and_mod(self):
    """Test 'import root-XXX.mod-XXX' remapping both to root3.mod3"""
    try:
        root3
    except NameError:
        pass
    else:
        self.fail('root3 was not supposed to exist yet')
    InstrumentedImportReplacer(scope=globals(), name='root3', module_path=[self.root_name], member=None, children={'mod3': ([self.root_name, self.mod_name], None, {})})
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root3, '__class__'))
    self.assertEqual(1, root3.var1)
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root3.mod3, '__class__'))
    self.assertEqual(2, root3.mod3.var2)
    mod_path = self.root_name + '.' + self.mod_name
    self.assertEqual([('__getattribute__', 'var1'), ('_import', 'root3'), ('import', self.root_name, [], 0), ('__getattribute__', 'var2'), ('_import', 'mod3'), ('import', mod_path, [], 0)], self.actions)