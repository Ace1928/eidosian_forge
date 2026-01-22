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
def clone_virtualenv(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        raise UserError('src dir %r does not exist' % src_dir)
    if os.path.exists(dst_dir):
        raise UserError('dest dir %r exists' % dst_dir)
    logger.info("cloning virtualenv '%s' => '%s'..." % (src_dir, dst_dir))
    shutil.copytree(src_dir, dst_dir, symlinks=True, ignore=shutil.ignore_patterns('*.pyc'))
    version, sys_path = _virtualenv_sys(dst_dir)
    logger.info('fixing scripts in bin...')
    fixup_scripts(src_dir, dst_dir, version)
    has_old = lambda s: any((i for i in s if _dirmatch(i, src_dir)))
    if has_old(sys_path):
        logger.info('fixing paths in sys.path...')
        fixup_syspath_items(sys_path, src_dir, dst_dir)
    v_sys = _virtualenv_sys(dst_dir)
    remaining = has_old(v_sys[1])
    assert not remaining, v_sys
    fix_symlink_if_necessary(src_dir, dst_dir)