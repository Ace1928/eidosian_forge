from distutils.dir_util import remove_tree, mkpath
from distutils import log
from types import CodeType
import sys
import os
import re
import textwrap
import marshal
from setuptools.extension import Library
from setuptools import Command
from .._path import ensure_directory
from sysconfig import get_path, get_python_version
def get_ext_outputs(self):
    """Get a list of relative paths to C extensions in the output distro"""
    all_outputs = []
    ext_outputs = []
    paths = {self.bdist_dir: ''}
    for base, dirs, files in sorted_walk(self.bdist_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in NATIVE_EXTENSIONS:
                all_outputs.append(paths[base] + filename)
        for filename in dirs:
            paths[os.path.join(base, filename)] = paths[base] + filename + '/'
    if self.distribution.has_ext_modules():
        build_cmd = self.get_finalized_command('build_ext')
        for ext in build_cmd.extensions:
            if isinstance(ext, Library):
                continue
            fullname = build_cmd.get_ext_fullname(ext.name)
            filename = build_cmd.get_ext_filename(fullname)
            if not os.path.basename(filename).startswith('dl-'):
                if os.path.exists(os.path.join(self.bdist_dir, filename)):
                    ext_outputs.append(filename)
    return (all_outputs, ext_outputs)