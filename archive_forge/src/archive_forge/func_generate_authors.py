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
def generate_authors(git_dir=None, dest_dir='.', option_dict=dict()):
    """Create AUTHORS file using git commits."""
    should_skip = options.get_boolean_option(option_dict, 'skip_authors', 'SKIP_GENERATE_AUTHORS')
    if should_skip:
        return
    start = time.time()
    old_authors = os.path.join(dest_dir, 'AUTHORS.in')
    new_authors = os.path.join(dest_dir, 'AUTHORS')
    if os.path.exists(new_authors) and (not os.access(new_authors, os.W_OK)):
        return
    log.info('[pbr] Generating AUTHORS')
    ignore_emails = '((jenkins|zuul)@review|infra@lists|jenkins@openstack)'
    if git_dir is None:
        git_dir = _get_git_directory()
    if git_dir:
        authors = []
        git_log_cmd = ['log', '--format=%aN <%aE>']
        authors += _run_git_command(git_log_cmd, git_dir).split('\n')
        authors = [a for a in authors if not re.search(ignore_emails, a)]
        co_authors_out = _run_git_command('log', git_dir)
        co_authors = re.findall('Co-authored-by:.+', co_authors_out, re.MULTILINE)
        co_authors = [signed.split(':', 1)[1].strip() for signed in co_authors if signed]
        authors += co_authors
        authors = sorted(set(authors))
        with open(new_authors, 'wb') as new_authors_fh:
            if os.path.exists(old_authors):
                with open(old_authors, 'rb') as old_authors_fh:
                    new_authors_fh.write(old_authors_fh.read())
            new_authors_fh.write(('\n'.join(authors) + '\n').encode('utf-8'))
    stop = time.time()
    log.info('[pbr] AUTHORS complete (%0.1fs)' % (stop - start))