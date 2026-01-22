import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_root_sub_submod(self):
    """Test import root.mod, root.sub.submoda, root.sub.submodb
        root should be a lazy import, with multiple children, who also
        have children to be imported.
        And when root is imported, the children should be lazy, and
        reuse the intermediate lazy object.
        """
    try:
        root5
    except NameError:
        pass
    else:
        self.fail('root5 was not supposed to exist yet')
    InstrumentedImportReplacer(scope=globals(), name='root5', module_path=[self.root_name], member=None, children={'mod5': ([self.root_name, self.mod_name], None, {}), 'sub5': ([self.root_name, self.sub_name], None, {'submoda5': ([self.root_name, self.sub_name, self.submoda_name], None, {}), 'submodb5': ([self.root_name, self.sub_name, self.submodb_name], None, {})})})
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5, '__class__'))
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.mod5, '__class__'))
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.sub5, '__class__'))
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.sub5.submoda5, '__class__'))
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.sub5.submodb5, '__class__'))
    self.assertEqual(2, root5.mod5.var2)
    self.assertEqual(4, root5.sub5.submoda5.var4)
    self.assertEqual(5, root5.sub5.submodb5.var5)
    mod_path = self.root_name + '.' + self.mod_name
    sub_path = self.root_name + '.' + self.sub_name
    submoda_path = sub_path + '.' + self.submoda_name
    submodb_path = sub_path + '.' + self.submodb_name
    self.assertEqual([('__getattribute__', 'mod5'), ('_import', 'root5'), ('import', self.root_name, [], 0), ('__getattribute__', 'submoda5'), ('_import', 'sub5'), ('import', sub_path, [], 0), ('__getattribute__', 'var2'), ('_import', 'mod5'), ('import', mod_path, [], 0), ('__getattribute__', 'var4'), ('_import', 'submoda5'), ('import', submoda_path, [], 0), ('__getattribute__', 'var5'), ('_import', 'submodb5'), ('import', submodb_path, [], 0)], self.actions)