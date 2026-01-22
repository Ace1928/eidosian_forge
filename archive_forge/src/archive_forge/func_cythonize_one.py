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
@record_results
def cythonize_one(pyx_file, c_file, fingerprint, quiet, options=None, raise_on_failure=True, embedded_metadata=None, full_module_name=None, show_all_warnings=False, progress=''):
    from ..Compiler.Main import compile_single, default_options
    from ..Compiler.Errors import CompileError, PyrexError
    if fingerprint:
        if not os.path.exists(options.cache):
            safe_makedirs(options.cache)
        fingerprint_file_base = join_path(options.cache, '%s-%s' % (os.path.basename(c_file), fingerprint))
        gz_fingerprint_file = fingerprint_file_base + gzip_ext
        zip_fingerprint_file = fingerprint_file_base + '.zip'
        if os.path.exists(gz_fingerprint_file) or os.path.exists(zip_fingerprint_file):
            if not quiet:
                print(u'%sFound compiled %s in cache' % (progress, pyx_file))
            if os.path.exists(gz_fingerprint_file):
                os.utime(gz_fingerprint_file, None)
                with contextlib.closing(gzip_open(gz_fingerprint_file, 'rb')) as g:
                    with contextlib.closing(open(c_file, 'wb')) as f:
                        shutil.copyfileobj(g, f)
            else:
                os.utime(zip_fingerprint_file, None)
                dirname = os.path.dirname(c_file)
                with contextlib.closing(zipfile.ZipFile(zip_fingerprint_file)) as z:
                    for artifact in z.namelist():
                        z.extract(artifact, os.path.join(dirname, artifact))
            return
    if not quiet:
        print(u'%sCythonizing %s' % (progress, Utils.decode_filename(pyx_file)))
    if options is None:
        options = CompilationOptions(default_options)
    options.output_file = c_file
    options.embedded_metadata = embedded_metadata
    old_warning_level = Errors.LEVEL
    if show_all_warnings:
        Errors.LEVEL = 0
    any_failures = 0
    try:
        result = compile_single(pyx_file, options, full_module_name=full_module_name)
        if result.num_errors > 0:
            any_failures = 1
    except (EnvironmentError, PyrexError) as e:
        sys.stderr.write('%s\n' % e)
        any_failures = 1
        import traceback
        traceback.print_exc()
    except Exception:
        if raise_on_failure:
            raise
        import traceback
        traceback.print_exc()
        any_failures = 1
    finally:
        if show_all_warnings:
            Errors.LEVEL = old_warning_level
    if any_failures:
        if raise_on_failure:
            raise CompileError(None, pyx_file)
        elif os.path.exists(c_file):
            os.remove(c_file)
    elif fingerprint:
        artifacts = list(filter(None, [getattr(result, attr, None) for attr in ('c_file', 'h_file', 'api_file', 'i_file')]))
        if len(artifacts) == 1:
            fingerprint_file = gz_fingerprint_file
            with contextlib.closing(open(c_file, 'rb')) as f:
                with contextlib.closing(gzip_open(fingerprint_file + '.tmp', 'wb')) as g:
                    shutil.copyfileobj(f, g)
        else:
            fingerprint_file = zip_fingerprint_file
            with contextlib.closing(zipfile.ZipFile(fingerprint_file + '.tmp', 'w', zipfile_compression_mode)) as zip:
                for artifact in artifacts:
                    zip.write(artifact, os.path.basename(artifact))
        os.rename(fingerprint_file + '.tmp', fingerprint_file)