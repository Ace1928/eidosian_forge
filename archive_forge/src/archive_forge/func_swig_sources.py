import os
import re
import sys
import shlex
import copy
from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError
from numpy.distutils import log
from numpy.distutils.misc_util import (
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file
def swig_sources(self, sources, extension):
    new_sources = []
    swig_sources = []
    swig_targets = {}
    target_dirs = []
    py_files = []
    target_ext = '.c'
    if '-c++' in extension.swig_opts:
        typ = 'c++'
        is_cpp = True
        extension.swig_opts.remove('-c++')
    elif self.swig_cpp:
        typ = 'c++'
        is_cpp = True
    else:
        typ = None
        is_cpp = False
    skip_swig = 0
    ext_name = extension.name.split('.')[-1]
    for source in sources:
        base, ext = os.path.splitext(source)
        if ext == '.i':
            if self.inplace:
                target_dir = os.path.dirname(base)
                py_target_dir = self.ext_target_dir
            else:
                target_dir = appendpath(self.build_src, os.path.dirname(base))
                py_target_dir = target_dir
            if os.path.isfile(source):
                name = get_swig_modulename(source)
                if name != ext_name[1:]:
                    raise DistutilsSetupError('mismatch of extension names: %s provides %r but expected %r' % (source, name, ext_name[1:]))
                if typ is None:
                    typ = get_swig_target(source)
                    is_cpp = typ == 'c++'
                else:
                    typ2 = get_swig_target(source)
                    if typ2 is None:
                        log.warn('source %r does not define swig target, assuming %s swig target' % (source, typ))
                    elif typ != typ2:
                        log.warn('expected %r but source %r defines %r swig target' % (typ, source, typ2))
                        if typ2 == 'c++':
                            log.warn('resetting swig target to c++ (some targets may have .c extension)')
                            is_cpp = True
                        else:
                            log.warn('assuming that %r has c++ swig target' % source)
                if is_cpp:
                    target_ext = '.cpp'
                target_file = os.path.join(target_dir, '%s_wrap%s' % (name, target_ext))
            else:
                log.warn("  source %s does not exist: skipping swig'ing." % source)
                name = ext_name[1:]
                skip_swig = 1
                target_file = _find_swig_target(target_dir, name)
                if not os.path.isfile(target_file):
                    log.warn('  target %s does not exist:\n   Assuming %s_wrap.{c,cpp} was generated with "build_src --inplace" command.' % (target_file, name))
                    target_dir = os.path.dirname(base)
                    target_file = _find_swig_target(target_dir, name)
                    if not os.path.isfile(target_file):
                        raise DistutilsSetupError('%r missing' % (target_file,))
                    log.warn('   Yes! Using %r as up-to-date target.' % target_file)
            target_dirs.append(target_dir)
            new_sources.append(target_file)
            py_files.append(os.path.join(py_target_dir, name + '.py'))
            swig_sources.append(source)
            swig_targets[source] = new_sources[-1]
        else:
            new_sources.append(source)
    if not swig_sources:
        return new_sources
    if skip_swig:
        return new_sources + py_files
    for d in target_dirs:
        self.mkpath(d)
    swig = self.swig or self.find_swig()
    swig_cmd = [swig, '-python'] + extension.swig_opts
    if is_cpp:
        swig_cmd.append('-c++')
    for d in extension.include_dirs:
        swig_cmd.append('-I' + d)
    for source in swig_sources:
        target = swig_targets[source]
        depends = [source] + extension.depends
        if self.force or newer_group(depends, target, 'newer'):
            log.info('%s: %s' % (os.path.basename(swig) + (is_cpp and '++' or ''), source))
            self.spawn(swig_cmd + self.swig_opts + ['-o', target, '-outdir', py_target_dir, source])
        else:
            log.debug("  skipping '%s' swig interface (up-to-date)" % source)
    return new_sources + py_files