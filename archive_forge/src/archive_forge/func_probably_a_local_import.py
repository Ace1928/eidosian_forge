from os.path import dirname, join, exists, sep
from lib2to3.fixes.fix_import import FixImport
from lib2to3.fixer_util import FromImport, syms
from lib2to3.fixes.fix_import import traverse_imports
from libfuturize.fixer_util import future_import
def probably_a_local_import(self, imp_name):
    """
        Like the corresponding method in the base class, but this also
        supports Cython modules.
        """
    if imp_name.startswith(u'.'):
        return False
    imp_name = imp_name.split(u'.', 1)[0]
    base_path = dirname(self.filename)
    base_path = join(base_path, imp_name)
    if not exists(join(dirname(base_path), '__init__.py')):
        return False
    for ext in ['.py', sep, '.pyc', '.so', '.sl', '.pyd', '.pyx']:
        if exists(base_path + ext):
            return True
    return False