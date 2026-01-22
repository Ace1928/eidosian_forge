from distutils.command.build_py import build_py as old_build_py
from numpy.distutils.misc_util import is_string
def find_package_modules(self, package, package_dir):
    modules = old_build_py.find_package_modules(self, package, package_dir)
    build_src = self.get_finalized_command('build_src')
    modules += build_src.py_modules_dict.get(package, [])
    return modules