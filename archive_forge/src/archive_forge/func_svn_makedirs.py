import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def svn_makedirs(dir, svn_add, verbosity, pad):
    parent = os.path.dirname(os.path.abspath(dir))
    if not os.path.exists(parent):
        svn_makedirs(parent, svn_add, verbosity, pad)
    os.mkdir(dir)
    if not svn_add:
        return
    if os.system('svn info %r >/dev/null 2>&1' % parent) > 0:
        if verbosity > 1:
            print('%sNot part of a svn working copy; cannot add directory' % pad)
        return
    cmd = ['svn', 'add', dir]
    if verbosity > 1:
        print('%sRunning: %s' % (pad, ' '.join(cmd)))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if verbosity > 1 and stdout:
        print('Script output:')
        print(stdout)