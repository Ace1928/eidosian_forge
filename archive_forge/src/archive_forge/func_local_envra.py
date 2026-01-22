from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
def local_envra(self, path):
    """return envra of a local rpm passed in"""
    ts = rpm.TransactionSet()
    ts.setVSFlags(rpm._RPMVSF_NOSIGNATURES)
    fd = os.open(path, os.O_RDONLY)
    try:
        header = ts.hdrFromFdno(fd)
    except rpm.error as e:
        return None
    finally:
        os.close(fd)
    return '%s:%s-%s-%s.%s' % (header[rpm.RPMTAG_EPOCH] or '0', header[rpm.RPMTAG_NAME], header[rpm.RPMTAG_VERSION], header[rpm.RPMTAG_RELEASE], header[rpm.RPMTAG_ARCH])