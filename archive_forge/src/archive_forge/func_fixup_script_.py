from __future__ import with_statement
import logging
import optparse
import os
import os.path
import re
import shutil
import subprocess
import sys
import itertools
def fixup_script_(root, file_, old_dir, new_dir, version, rewrite_env_python=False):
    old_shebang = '#!%s/bin/python' % os.path.normcase(os.path.abspath(old_dir))
    new_shebang = '#!%s/bin/python' % os.path.normcase(os.path.abspath(new_dir))
    env_shebang = '#!/usr/bin/env python'
    filename = os.path.join(root, file_)
    with open(filename, 'rb') as f:
        if f.read(2) != b'#!':
            return
        f.seek(0)
        lines = f.readlines()
    if not lines:
        return

    def rewrite_shebang(version=None):
        logger.debug('fixing %s' % filename)
        shebang = new_shebang
        if version:
            shebang = shebang + version
        shebang = (shebang + '\n').encode('utf-8')
        with open(filename, 'wb') as f:
            f.write(shebang)
            f.writelines(lines[1:])
    try:
        bang = lines[0].decode('utf-8').strip()
    except UnicodeDecodeError:
        return
    short_version = bang[len(old_shebang):]
    if not bang.startswith('#!'):
        return
    elif bang == old_shebang:
        rewrite_shebang()
    elif bang.startswith(old_shebang) and bang[len(old_shebang):] == version:
        rewrite_shebang(version)
    elif bang.startswith(old_shebang) and short_version and (bang[len(old_shebang):] == short_version):
        rewrite_shebang(short_version)
    elif rewrite_env_python and bang.startswith(env_shebang):
        if bang == env_shebang:
            rewrite_shebang()
        elif bang[len(env_shebang):] == version:
            rewrite_shebang(version)
    else:
        return