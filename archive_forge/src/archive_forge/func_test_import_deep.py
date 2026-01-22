import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_deep(self):
    """Test import root.mod, root.sub.submoda, root.sub.submodb
        root should be a lazy import, with multiple children, who also
        have children to be imported.
        And when root is imported, the children should be lazy, and
        reuse the intermediate lazy object.
        """
    try:
        submoda7
    except NameError:
        pass
    else:
        self.fail('submoda7 was not supposed to exist yet')
    text = 'import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7\n' % self.__dict__
    proc = lazy_import.ImportProcessor(InstrumentedImportReplacer)
    proc.lazy_import(scope=globals(), text=text)
    self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(submoda7, '__class__'))
    self.assertEqual(4, submoda7.var4)
    sub_path = self.root_name + '.' + self.sub_name
    submoda_path = sub_path + '.' + self.submoda_name
    self.assertEqual([('__getattribute__', 'var4'), ('_import', 'submoda7'), ('import', submoda_path, [], 0)], self.actions)