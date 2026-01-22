from __future__ import absolute_import, print_function
import os
import shutil
import tempfile
from .Dependencies import cythonize, extended_iglob
from ..Utils import is_package_dir
from ..Compiler import Options
def create_args_parser():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from ..Compiler.CmdLine import ParseDirectivesAction, ParseOptionsAction, ParseCompileTimeEnvAction
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, epilog='Environment variables:\n  CYTHON_FORCE_REGEN: if set to 1, forces cythonize to regenerate the output files regardless\n        of modification times and changes.\n  Environment variables accepted by setuptools are supported to configure the C compiler and build:\n  https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#compiler-and-linker-options')
    parser.add_argument('-X', '--directive', metavar='NAME=VALUE,...', dest='directives', default={}, type=str, action=ParseDirectivesAction, help='set a compiler directive')
    parser.add_argument('-E', '--compile-time-env', metavar='NAME=VALUE,...', dest='compile_time_env', default={}, type=str, action=ParseCompileTimeEnvAction, help='set a compile time environment variable')
    parser.add_argument('-s', '--option', metavar='NAME=VALUE', dest='options', default={}, type=str, action=ParseOptionsAction, help='set a cythonize option')
    parser.add_argument('-2', dest='language_level', action='store_const', const=2, default=None, help='use Python 2 syntax mode by default')
    parser.add_argument('-3', dest='language_level', action='store_const', const=3, help='use Python 3 syntax mode by default')
    parser.add_argument('--3str', dest='language_level', action='store_const', const='3str', help='use Python 3 syntax mode by default')
    parser.add_argument('-+', '--cplus', dest='language', action='store_const', const='c++', default=None, help='Compile as C++ rather than C')
    parser.add_argument('-a', '--annotate', action='store_const', const='default', dest='annotate', help='Produce a colorized HTML version of the source.')
    parser.add_argument('--annotate-fullc', action='store_const', const='fullc', dest='annotate', help='Produce a colorized HTML version of the source which includes entire generated C/C++-code.')
    parser.add_argument('-x', '--exclude', metavar='PATTERN', dest='excludes', action='append', default=[], help='exclude certain file patterns from the compilation')
    parser.add_argument('-b', '--build', dest='build', action='store_true', default=None, help='build extension modules using distutils/setuptools')
    parser.add_argument('-i', '--inplace', dest='build_inplace', action='store_true', default=None, help='build extension modules in place using distutils/setuptools (implies -b)')
    parser.add_argument('-j', '--parallel', dest='parallel', metavar='N', type=int, default=parallel_compiles, help='run builds in N parallel jobs (default: %d)' % parallel_compiles or 1)
    parser.add_argument('-f', '--force', dest='force', action='store_true', default=None, help='force recompilation')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=None, help='be less verbose during compilation')
    parser.add_argument('--lenient', dest='lenient', action='store_true', default=None, help='increase Python compatibility by ignoring some compile time errors')
    parser.add_argument('-k', '--keep-going', dest='keep_going', action='store_true', default=None, help='compile as much as possible, ignore compilation failures')
    parser.add_argument('--no-docstrings', dest='no_docstrings', action='store_true', default=None, help='strip docstrings')
    parser.add_argument('-M', '--depfile', action='store_true', help='produce depfiles for the sources')
    parser.add_argument('sources', nargs='*')
    return parser