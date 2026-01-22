from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
def cythonize(module_list, exclude=None, nthreads=0, aliases=None, quiet=False, force=None, language=None, exclude_failures=False, show_all_warnings=False, **options):
    """
    Compile a set of source modules into C/C++ files and return a list of distutils
    Extension objects for them.

    :param module_list: As module list, pass either a glob pattern, a list of glob
                        patterns or a list of Extension objects.  The latter
                        allows you to configure the extensions separately
                        through the normal distutils options.
                        You can also pass Extension objects that have
                        glob patterns as their sources. Then, cythonize
                        will resolve the pattern and create a
                        copy of the Extension for every matching file.

    :param exclude: When passing glob patterns as ``module_list``, you can exclude certain
                    module names explicitly by passing them into the ``exclude`` option.

    :param nthreads: The number of concurrent builds for parallel compilation
                     (requires the ``multiprocessing`` module).

    :param aliases: If you want to use compiler directives like ``# distutils: ...`` but
                    can only know at compile time (when running the ``setup.py``) which values
                    to use, you can use aliases and pass a dictionary mapping those aliases
                    to Python strings when calling :func:`cythonize`. As an example, say you
                    want to use the compiler
                    directive ``# distutils: include_dirs = ../static_libs/include/``
                    but this path isn't always fixed and you want to find it when running
                    the ``setup.py``. You can then do ``# distutils: include_dirs = MY_HEADERS``,
                    find the value of ``MY_HEADERS`` in the ``setup.py``, put it in a python
                    variable called ``foo`` as a string, and then call
                    ``cythonize(..., aliases={'MY_HEADERS': foo})``.

    :param quiet: If True, Cython won't print error, warning, or status messages during the
                  compilation.

    :param force: Forces the recompilation of the Cython modules, even if the timestamps
                  don't indicate that a recompilation is necessary.

    :param language: To globally enable C++ mode, you can pass ``language='c++'``. Otherwise, this
                     will be determined at a per-file level based on compiler directives.  This
                     affects only modules found based on file names.  Extension instances passed
                     into :func:`cythonize` will not be changed. It is recommended to rather
                     use the compiler directive ``# distutils: language = c++`` than this option.

    :param exclude_failures: For a broad 'try to compile' mode that ignores compilation
                             failures and simply excludes the failed extensions,
                             pass ``exclude_failures=True``. Note that this only
                             really makes sense for compiling ``.py`` files which can also
                             be used without compilation.

    :param show_all_warnings: By default, not all Cython warnings are printed.
                              Set to true to show all warnings.

    :param annotate: If ``True``, will produce a HTML file for each of the ``.pyx`` or ``.py``
                     files compiled. The HTML file gives an indication
                     of how much Python interaction there is in
                     each of the source code lines, compared to plain C code.
                     It also allows you to see the C/C++ code
                     generated for each line of Cython code. This report is invaluable when
                     optimizing a function for speed,
                     and for determining when to :ref:`release the GIL <nogil>`:
                     in general, a ``nogil`` block may contain only "white" code.
                     See examples in :ref:`determining_where_to_add_types` or
                     :ref:`primes`.


    :param annotate-fullc: If ``True`` will produce a colorized HTML version of
                           the source which includes entire generated C/C++-code.


    :param compiler_directives: Allow to set compiler directives in the ``setup.py`` like this:
                                ``compiler_directives={'embedsignature': True}``.
                                See :ref:`compiler-directives`.

    :param depfile: produce depfiles for the sources if True.
    """
    if exclude is None:
        exclude = []
    if 'include_path' not in options:
        options['include_path'] = ['.']
    if 'common_utility_include_dir' in options:
        safe_makedirs(options['common_utility_include_dir'])
    depfile = options.pop('depfile', None)
    if pythran is None:
        pythran_options = None
    else:
        pythran_options = CompilationOptions(**options)
        pythran_options.cplus = True
        pythran_options.np_pythran = True
    if force is None:
        force = os.environ.get('CYTHON_FORCE_REGEN') == '1'
    c_options = CompilationOptions(**options)
    cpp_options = CompilationOptions(**options)
    cpp_options.cplus = True
    ctx = Context.from_options(c_options)
    options = c_options
    module_list, module_metadata = create_extension_list(module_list, exclude=exclude, ctx=ctx, quiet=quiet, exclude_failures=exclude_failures, language=language, aliases=aliases)
    fix_windows_unicode_modules(module_list)
    deps = create_dependency_tree(ctx, quiet=quiet)
    build_dir = getattr(options, 'build_dir', None)

    def copy_to_build_dir(filepath, root=os.getcwd()):
        filepath_abs = os.path.abspath(filepath)
        if os.path.isabs(filepath):
            filepath = filepath_abs
        if filepath_abs.startswith(root):
            mod_dir = join_path(build_dir, os.path.dirname(_relpath(filepath, root)))
            copy_once_if_newer(filepath_abs, mod_dir)
    modules_by_cfile = collections.defaultdict(list)
    to_compile = []
    for m in module_list:
        if build_dir:
            for dep in m.depends:
                copy_to_build_dir(dep)
        cy_sources = [source for source in m.sources if os.path.splitext(source)[1] in ('.pyx', '.py')]
        if len(cy_sources) == 1:
            full_module_name = m.name
        else:
            full_module_name = None
        new_sources = []
        for source in m.sources:
            base, ext = os.path.splitext(source)
            if ext in ('.pyx', '.py'):
                if m.np_pythran:
                    c_file = base + '.cpp'
                    options = pythran_options
                elif m.language == 'c++':
                    c_file = base + '.cpp'
                    options = cpp_options
                else:
                    c_file = base + '.c'
                    options = c_options
                if build_dir:
                    if os.path.isabs(c_file):
                        c_file = os.path.splitdrive(c_file)[1]
                        c_file = c_file.split(os.sep, 1)[1]
                    c_file = os.path.join(build_dir, c_file)
                    dir = os.path.dirname(c_file)
                    safe_makedirs_once(dir)
                if depfile:
                    dependencies = deps.all_dependencies(source)
                    write_depfile(c_file, source, dependencies)
                if Utils.file_generated_by_this_cython(c_file):
                    c_timestamp = os.path.getmtime(c_file)
                else:
                    c_timestamp = -1
                if c_timestamp < deps.timestamp(source):
                    dep_timestamp, dep = (deps.timestamp(source), source)
                    priority = 0
                else:
                    dep_timestamp, dep = deps.newest_dependency(source)
                    priority = 2 - (dep in deps.immediate_dependencies(source))
                if force or c_timestamp < dep_timestamp:
                    if not quiet and (not force):
                        if source == dep:
                            print(u'Compiling %s because it changed.' % Utils.decode_filename(source))
                        else:
                            print(u'Compiling %s because it depends on %s.' % (Utils.decode_filename(source), Utils.decode_filename(dep)))
                    if not force and options.cache:
                        fingerprint = deps.transitive_fingerprint(source, m, options)
                    else:
                        fingerprint = None
                    to_compile.append((priority, source, c_file, fingerprint, quiet, options, not exclude_failures, module_metadata.get(m.name), full_module_name, show_all_warnings))
                new_sources.append(c_file)
                modules_by_cfile[c_file].append(m)
            else:
                new_sources.append(source)
                if build_dir:
                    copy_to_build_dir(source)
        m.sources = new_sources
    if options.cache:
        if not os.path.exists(options.cache):
            os.makedirs(options.cache)
    to_compile.sort()
    N = len(to_compile)
    progress_fmt = '[{0:%d}/{1}] ' % len(str(N))
    for i in range(N):
        progress = progress_fmt.format(i + 1, N)
        to_compile[i] = to_compile[i][1:] + (progress,)
    if N <= 1:
        nthreads = 0
    if nthreads:
        import multiprocessing
        pool = multiprocessing.Pool(nthreads, initializer=_init_multiprocessing_helper)
        try:
            result = pool.map_async(cythonize_one_helper, to_compile, chunksize=1)
            pool.close()
            while not result.ready():
                try:
                    result.get(99999)
                except multiprocessing.TimeoutError:
                    pass
        except KeyboardInterrupt:
            pool.terminate()
            raise
        pool.join()
    else:
        for args in to_compile:
            cythonize_one(*args)
    if exclude_failures:
        failed_modules = set()
        for c_file, modules in modules_by_cfile.items():
            if not os.path.exists(c_file):
                failed_modules.update(modules)
            elif os.path.getsize(c_file) < 200:
                f = io_open(c_file, 'r', encoding='iso8859-1')
                try:
                    if f.read(len('#error ')) == '#error ':
                        failed_modules.update(modules)
                finally:
                    f.close()
        if failed_modules:
            for module in failed_modules:
                module_list.remove(module)
            print(u'Failed compilations: %s' % ', '.join(sorted([module.name for module in failed_modules])))
    if options.cache:
        cleanup_cache(options.cache, getattr(options, 'cache_size', 1024 * 1024 * 100))
    sys.stdout.flush()
    return module_list