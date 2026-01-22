from __future__ import absolute_import, print_function
import io
import os
import re
import sys
import time
import copy
import distutils.log
import textwrap
import hashlib
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from IPython.core import display
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.utils.text import dedent
from ..Shadow import __version__ as cython_version
from ..Compiler.Errors import CompileError
from .Inline import cython_inline, load_dynamic
from .Dependencies import cythonize
from ..Utils import captured_fd, print_captured
@magic_arguments.magic_arguments()
@magic_arguments.argument('-a', '--annotate', action='store_const', const='default', dest='annotate', help='Produce a colorized HTML version of the source.')
@magic_arguments.argument('--annotate-fullc', action='store_const', const='fullc', dest='annotate', help='Produce a colorized HTML version of the source which includes entire generated C/C++-code.')
@magic_arguments.argument('-+', '--cplus', action='store_true', default=False, help='Output a C++ rather than C file.')
@magic_arguments.argument('-3', dest='language_level', action='store_const', const=3, default=None, help='Select Python 3 syntax.')
@magic_arguments.argument('-2', dest='language_level', action='store_const', const=2, default=None, help='Select Python 2 syntax.')
@magic_arguments.argument('-f', '--force', action='store_true', default=False, help='Force the compilation of a new module, even if the source has been previously compiled.')
@magic_arguments.argument('-c', '--compile-args', action='append', default=[], help='Extra flags to pass to compiler via the `extra_compile_args` Extension flag (can be specified  multiple times).')
@magic_arguments.argument('--link-args', action='append', default=[], help='Extra flags to pass to linker via the `extra_link_args` Extension flag (can be specified  multiple times).')
@magic_arguments.argument('-l', '--lib', action='append', default=[], help='Add a library to link the extension against (can be specified multiple times).')
@magic_arguments.argument('-n', '--name', help='Specify a name for the Cython module.')
@magic_arguments.argument('-L', dest='library_dirs', metavar='dir', action='append', default=[], help='Add a path to the list of library directories (can be specified multiple times).')
@magic_arguments.argument('-I', '--include', action='append', default=[], help='Add a path to the list of include directories (can be specified multiple times).')
@magic_arguments.argument('-S', '--src', action='append', default=[], help='Add a path to the list of src files (can be specified multiple times).')
@magic_arguments.argument('--pgo', dest='pgo', action='store_true', default=False, help='Enable profile guided optimisation in the C compiler. Compiles the cell twice and executes it in between to generate a runtime profile.')
@magic_arguments.argument('--verbose', dest='quiet', action='store_false', default=True, help='Print debug information like generated .c/.cpp file location and exact gcc/g++ command invoked.')
@cell_magic
def cython(self, line, cell):
    """Compile and import everything from a Cython code cell.

        The contents of the cell are written to a `.pyx` file in the
        directory `IPYTHONDIR/cython` using a filename with the hash of the
        code. This file is then cythonized and compiled. The resulting module
        is imported and all of its symbols are injected into the user's
        namespace. The usage is similar to that of `%%cython_pyximport` but
        you don't have to pass a module name::

            %%cython
            def f(x):
                return 2.0*x

        To compile OpenMP codes, pass the required  `--compile-args`
        and `--link-args`.  For example with gcc::

            %%cython --compile-args=-fopenmp --link-args=-fopenmp
            ...

        To enable profile guided optimisation, pass the ``--pgo`` option.
        Note that the cell itself needs to take care of establishing a suitable
        profile when executed. This can be done by implementing the functions to
        optimise, and then calling them directly in the same cell on some realistic
        training data like this::

            %%cython --pgo
            def critical_function(data):
                for item in data:
                    ...

            # execute function several times to build profile
            from somewhere import some_typical_data
            for _ in range(100):
                critical_function(some_typical_data)

        In Python 3.5 and later, you can distinguish between the profile and
        non-profile runs as follows::

            if "_pgo_" in __name__:
                ...  # execute critical code here
        """
    args = magic_arguments.parse_argstring(self.cython, line)
    code = cell if cell.endswith('\n') else cell + '\n'
    lib_dir = os.path.join(get_ipython_cache_dir(), 'cython')
    key = (code, line, sys.version_info, sys.executable, cython_version)
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
    if args.pgo:
        key += ('pgo',)
    if args.force:
        key += (time.time(),)
    if args.name:
        module_name = str(args.name)
    else:
        module_name = '_cython_magic_' + hashlib.sha1(str(key).encode('utf-8')).hexdigest()
    html_file = os.path.join(lib_dir, module_name + '.html')
    module_path = os.path.join(lib_dir, module_name + self.so_ext)
    have_module = os.path.isfile(module_path)
    need_cythonize = args.pgo or not have_module
    if args.annotate:
        if not os.path.isfile(html_file):
            need_cythonize = True
    extension = None
    if need_cythonize:
        extensions = self._cythonize(module_name, code, lib_dir, args, quiet=args.quiet)
        if extensions is None:
            return None
        assert len(extensions) == 1
        extension = extensions[0]
        self._code_cache[key] = module_name
        if args.pgo:
            self._profile_pgo_wrapper(extension, lib_dir)

    def print_compiler_output(stdout, stderr, where):
        print_captured(stdout, where, u'Content of stdout:\n')
        print_captured(stderr, where, u'Content of stderr:\n')
    get_stderr = get_stdout = None
    try:
        with captured_fd(1) as get_stdout:
            with captured_fd(2) as get_stderr:
                self._build_extension(extension, lib_dir, pgo_step_name='use' if args.pgo else None, quiet=args.quiet)
    except (distutils.errors.CompileError, distutils.errors.LinkError):
        print_compiler_output(get_stdout(), get_stderr(), sys.stderr)
        return None
    print_compiler_output(get_stdout(), get_stderr(), sys.stdout)
    module = load_dynamic(module_name, module_path)
    self._import_all(module)
    if args.annotate:
        try:
            with io.open(html_file, encoding='utf-8') as f:
                annotated_html = f.read()
        except IOError as e:
            print('Cython completed successfully but the annotated source could not be read.', file=sys.stderr)
            print(e, file=sys.stderr)
        else:
            return display.HTML(self.clean_annotated_html(annotated_html))