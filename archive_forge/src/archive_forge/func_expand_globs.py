import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
def expand_globs(self):
    finished = []
    for line in self.data_files.split('\n'):
        if line.rstrip().endswith('*') and '=' in line:
            target, source_glob = line.split('=')
            source_prefix = source_glob.strip()[:-1]
            target = target.strip()
            if not target.endswith(os.path.sep):
                target += os.path.sep
            unquoted_prefix = unquote_path(source_prefix)
            unquoted_target = unquote_path(target)
            for dirpath, dirnames, fnames in os.walk(unquoted_prefix):
                new_prefix = dirpath.replace(unquoted_prefix, unquoted_target, 1)
                finished.append("'%s' = " % new_prefix)
                finished.extend([" '%s'" % os.path.join(dirpath, f) for f in fnames])
        else:
            finished.append(line)
    self.data_files = '\n'.join(finished)