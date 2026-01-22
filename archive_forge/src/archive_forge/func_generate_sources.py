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
def generate_sources(self, sources, extension):
    new_sources = []
    func_sources = []
    for source in sources:
        if is_string(source):
            new_sources.append(source)
        else:
            func_sources.append(source)
    if not func_sources:
        return new_sources
    if self.inplace and (not is_sequence(extension)):
        build_dir = self.ext_target_dir
    else:
        if is_sequence(extension):
            name = extension[0]
        else:
            name = extension.name
        build_dir = os.path.join(*[self.build_src] + name.split('.')[:-1])
    self.mkpath(build_dir)
    if self.verbose_cfg:
        new_level = log.INFO
    else:
        new_level = log.WARN
    old_level = log.set_threshold(new_level)
    for func in func_sources:
        source = func(extension, build_dir)
        if not source:
            continue
        if is_sequence(source):
            [log.info("  adding '%s' to sources." % (s,)) for s in source]
            new_sources.extend(source)
        else:
            log.info("  adding '%s' to sources." % (source,))
            new_sources.append(source)
    log.set_threshold(old_level)
    return new_sources