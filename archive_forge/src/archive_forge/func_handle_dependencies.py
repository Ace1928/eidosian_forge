imports, including parts of the standard library and installed
import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
def handle_dependencies(pyxfilename):
    testing = '_test_files' in globals()
    dependfile = os.path.splitext(pyxfilename)[0] + PYXDEP_EXT
    if os.path.exists(dependfile):
        with open(dependfile) as fid:
            depends = fid.readlines()
        depends = [depend.strip() for depend in depends]
        files = [dependfile]
        for depend in depends:
            fullpath = os.path.join(os.path.dirname(dependfile), depend)
            files.extend(glob.glob(fullpath))
        if testing:
            _test_files[:] = []
        for file in files:
            from distutils.dep_util import newer
            if newer(file, pyxfilename):
                _debug('Rebuilding %s because of %s', pyxfilename, file)
                filetime = os.path.getmtime(file)
                os.utime(pyxfilename, (filetime, filetime))
                if testing:
                    _test_files.append(file)