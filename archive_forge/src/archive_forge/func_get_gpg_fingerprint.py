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
def get_gpg_fingerprint(output):
    """Return a fingerprint of the primary key.

    Ref:
    https://git.gnupg.org/cgi-bin/gitweb.cgi?p=gnupg.git;a=blob;f=doc/DETAILS;hb=HEAD#l482
    """
    for line in output.splitlines():
        data = line.split()
        if data[1] != 'VALIDSIG':
            continue
        data_id = 11 if len(data) == 11 else 2
        return data[data_id]