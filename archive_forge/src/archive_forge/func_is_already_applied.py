from __future__ import absolute_import, division, print_function
import os
import platform
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def is_already_applied(patch_func, patch_file, basedir, dest_file=None, binary=False, ignore_whitespace=False, strip=0, state='present'):
    opts = ['--quiet', '--forward', '--strip=%s' % strip, "--directory='%s'" % basedir, "--input='%s'" % patch_file]
    add_dry_run_option(opts)
    if binary:
        opts.append('--binary')
    if ignore_whitespace:
        opts.append('--ignore-whitespace')
    if dest_file:
        opts.append("'%s'" % dest_file)
    if state == 'present':
        opts.append('--reverse')
    rc, var1, var2 = patch_func(opts)
    return rc == 0