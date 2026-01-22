from __future__ import unicode_literals
import distutils.errors
from distutils import log
import errno
import io
import os
import re
import subprocess
import time
import pkg_resources
from pbr import options
from pbr import version
def write_git_changelog(git_dir=None, dest_dir=os.path.curdir, option_dict=None, changelog=None):
    """Write a changelog based on the git changelog."""
    start = time.time()
    if not option_dict:
        option_dict = {}
    should_skip = options.get_boolean_option(option_dict, 'skip_changelog', 'SKIP_WRITE_GIT_CHANGELOG')
    if should_skip:
        return
    if not changelog:
        changelog = _iter_log_oneline(git_dir=git_dir)
        if changelog:
            changelog = _iter_changelog(changelog)
    if not changelog:
        return
    new_changelog = os.path.join(dest_dir, 'ChangeLog')
    if os.path.exists(new_changelog) and (not os.access(new_changelog, os.W_OK)):
        log.info('[pbr] ChangeLog not written (file already exists and it is not writeable)')
        return
    log.info('[pbr] Writing ChangeLog')
    with io.open(new_changelog, 'w', encoding='utf-8') as changelog_file:
        for release, content in changelog:
            changelog_file.write(content)
    stop = time.time()
    log.info('[pbr] ChangeLog complete (%0.1fs)' % (stop - start))