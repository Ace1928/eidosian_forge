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
def get_annotated_tags(git_path, module, dest):
    tags = []
    cmd = [git_path, 'for-each-ref', 'refs/tags/', '--format', '%(objecttype):%(refname:short)']
    rc, out, err = module.run_command(cmd, cwd=dest)
    if rc != 0:
        module.fail_json(msg='Could not determine tag data - received %s' % out, stdout=out, stderr=err)
    for line in to_native(out).split('\n'):
        if line.strip():
            tagtype, tagname = line.strip().split(':')
            if tagtype == 'tag':
                tags.append(tagname)
    return tags