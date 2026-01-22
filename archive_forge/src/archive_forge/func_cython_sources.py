import sys
import os
from distutils.errors import DistutilsPlatformError
from distutils.dep_util import newer, newer_group
from distutils import log
from distutils.command import build_ext as _build_ext
from distutils import sysconfig
import inspect
import warnings
def cython_sources(self, sources, extension):
    """
        Walk the list of source files in 'sources', looking for Cython
        source files (.pyx and .py).  Run Cython on all that are
        found, and return a modified 'sources' list with Cython source
        files replaced by the generated C (or C++) files.
        """
    new_sources = []
    cython_sources = []
    cython_targets = {}
    create_listing = self.cython_create_listing or getattr(extension, 'cython_create_listing', 0)
    line_directives = self.cython_line_directives or getattr(extension, 'cython_line_directives', 0)
    no_c_in_traceback = self.no_c_in_traceback or getattr(extension, 'no_c_in_traceback', 0)
    cplus = self.cython_cplus or getattr(extension, 'cython_cplus', 0) or (extension.language and extension.language.lower() == 'c++')
    cython_gen_pxi = self.cython_gen_pxi or getattr(extension, 'cython_gen_pxi', 0)
    cython_gdb = self.cython_gdb or getattr(extension, 'cython_gdb', False)
    cython_compile_time_env = self.cython_compile_time_env or getattr(extension, 'cython_compile_time_env', None)
    includes = list(self.cython_include_dirs)
    try:
        for i in extension.cython_include_dirs:
            if i not in includes:
                includes.append(i)
    except AttributeError:
        pass
    extension.include_dirs = list(extension.include_dirs)
    for i in extension.include_dirs:
        if i not in includes:
            includes.append(i)
    directives = dict(self.cython_directives)
    if hasattr(extension, 'cython_directives'):
        directives.update(extension.cython_directives)
    if cplus:
        target_ext = '.cpp'
    else:
        target_ext = '.c'
    if not self.inplace and (self.cython_c_in_temp or getattr(extension, 'cython_c_in_temp', 0)):
        target_dir = os.path.join(self.build_temp, 'pyrex')
        for package_name in extension.name.split('.')[:-1]:
            target_dir = os.path.join(target_dir, package_name)
    else:
        target_dir = None
    newest_dependency = None
    for source in sources:
        base, ext = os.path.splitext(os.path.basename(source))
        if ext == '.py':
            ext = '.pyx'
        if ext == '.pyx':
            output_dir = target_dir or os.path.dirname(source)
            new_sources.append(os.path.join(output_dir, base + target_ext))
            cython_sources.append(source)
            cython_targets[source] = new_sources[-1]
        elif ext == '.pxi' or ext == '.pxd':
            if newest_dependency is None or newer(source, newest_dependency):
                newest_dependency = source
        else:
            new_sources.append(source)
    if not cython_sources:
        return new_sources
    try:
        from Cython.Compiler.Main import CompilationOptions, default_options as cython_default_options, compile as cython_compile
        from Cython.Compiler.Errors import PyrexError
    except ImportError:
        e = sys.exc_info()[1]
        print('failed to import Cython: %s' % e)
        raise DistutilsPlatformError('Cython does not appear to be installed')
    module_name = extension.name
    for source in cython_sources:
        target = cython_targets[source]
        depends = [source] + list(extension.depends or ())
        if source[-4:].lower() == '.pyx' and os.path.isfile(source[:-3] + 'pxd'):
            depends += [source[:-3] + 'pxd']
        rebuild = self.force or newer_group(depends, target, 'newer')
        if not rebuild and newest_dependency is not None:
            rebuild = newer(newest_dependency, target)
        if rebuild:
            log.info('cythoning %s to %s', source, target)
            self.mkpath(os.path.dirname(target))
            if self.inplace:
                output_dir = os.curdir
            else:
                output_dir = self.build_lib
            options = CompilationOptions(cython_default_options, use_listing_file=create_listing, include_path=includes, compiler_directives=directives, output_file=target, cplus=cplus, emit_linenums=line_directives, c_line_in_traceback=not no_c_in_traceback, generate_pxi=cython_gen_pxi, output_dir=output_dir, gdb_debug=cython_gdb, compile_time_env=cython_compile_time_env)
            result = cython_compile(source, options=options, full_module_name=module_name)
        else:
            log.info("skipping '%s' Cython extension (up-to-date)", target)
    return new_sources