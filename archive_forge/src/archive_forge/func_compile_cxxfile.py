from pythran.backend import Cxx, Python
from pythran.config import cfg
from pythran.cxxgen import PythonModule, Include, Line, Statement
from pythran.cxxgen import FunctionBody, FunctionDeclaration, Value, Block
from pythran.cxxgen import ReturnStatement
from pythran.dist import PythranExtension, PythranBuildExt
from pythran.middlend import refine, mark_unexported_functions
from pythran.passmanager import PassManager
from pythran.tables import pythran_ward
from pythran.types import tog
from pythran.types.type_dependencies import pytype_to_deps
from pythran.types.conversion import pytype_to_ctype
from pythran.spec import load_specfile, Spec
from pythran.spec import spec_to_string
from pythran.syntax import check_specs, check_exports, PythranSyntaxError
from pythran.version import __version__
from pythran.utils import cxxid
import pythran.frontend as frontend
from tempfile import mkdtemp, NamedTemporaryFile
import gast as ast
import importlib
import logging
import os.path
import shutil
import glob
import hashlib
from functools import reduce
import sys
def compile_cxxfile(module_name, cxxfile, output_binary=None, **kwargs):
    """c++ file -> native module
    Return the filename of the produced shared library
    Raises CompileError on failure

    """
    builddir = mkdtemp()
    buildtmp = mkdtemp()
    extension = PythranExtension(module_name, [cxxfile], **kwargs)
    try:
        setup(name=module_name, ext_modules=[extension], cmdclass={'build_ext': PythranBuildExt}, script_name='setup.py', script_args=['--verbose' if logger.isEnabledFor(logging.INFO) else '--quiet', 'build_ext', '--build-lib', builddir, '--build-temp', buildtmp])
    except SystemExit as e:
        raise CompileError(str(e))

    def copy(src_file, dest_file):
        with open(src_file, 'rb') as src:
            with open(dest_file, 'wb') as dest:
                dest.write(src.read())
    ext = sysconfig.get_config_var('EXT_SUFFIX')
    for f in glob.glob(os.path.join(builddir, module_name + '*')):
        if f.endswith(ext):
            if output_binary:
                output_binary = output_binary.replace('%{ext}', ext)
            else:
                output_binary = os.path.join(os.getcwd(), module_name + ext)
            copy(f, output_binary)
        else:
            if output_binary:
                output_binary = output_binary.replace('%{ext}', '')
                output_directory = os.path.dirname(output_binary)
            else:
                output_directory = os.getcwd()
            copy(f, os.path.join(output_directory, os.path.basename(f)))
    shutil.rmtree(builddir)
    shutil.rmtree(buildtmp)
    logger.info('Generated module: ' + module_name)
    logger.info('Output: ' + output_binary)
    return output_binary