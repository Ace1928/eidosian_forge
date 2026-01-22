from __future__ import absolute_import, print_function
import os
import shutil
import tempfile
from .Dependencies import cythonize, extended_iglob
from ..Utils import is_package_dir
from ..Compiler import Options
def find_package_base(path):
    base_dir, package_path = os.path.split(path)
    while is_package_dir(base_dir):
        base_dir, parent = os.path.split(base_dir)
        package_path = '%s/%s' % (parent, package_path)
    return (base_dir, package_path)