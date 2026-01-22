from __future__ import absolute_import, division, print_function
import filecmp
import os
import re
import shlex
import stat
import sys
import shutil
import tempfile
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.six import b, string_types
def head_splitter(headfile, remote, module=None, fail_on_error=False):
    """Extract the head reference"""
    res = None
    if os.path.exists(headfile):
        rawdata = None
        try:
            f = open(headfile, 'r')
            rawdata = f.readline()
            f.close()
        except Exception:
            if fail_on_error and module:
                module.fail_json(msg='Unable to read %s' % headfile)
        if rawdata:
            try:
                rawdata = rawdata.replace('refs/remotes/%s' % remote, '', 1)
                refparts = rawdata.split(' ')
                newref = refparts[-1]
                nrefparts = newref.split('/', 2)
                res = nrefparts[-1].rstrip('\n')
            except Exception:
                if fail_on_error and module:
                    module.fail_json(msg="Unable to split head from '%s'" % rawdata)
    return res