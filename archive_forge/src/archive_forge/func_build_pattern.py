from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, attr_chain
from lib2to3.fixes.fix_imports import alternates, build_pattern, FixImports
def build_pattern(self):
    return '|'.join(build_pattern(self.mapping))