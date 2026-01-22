from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import KatelloAnsibleModule, missing_required_lib
def get_rpm_info(path):
    ts = rpm.TransactionSet()
    try:
        vsflags = rpm.RPMVSF_MASK_NOSIGNATURES
    except AttributeError:
        vsflags = rpm._RPMVSF_NOSIGNATURES
    ts.setVSFlags(vsflags)
    with open(path) as rpmfile:
        rpmhdr = ts.hdrFromFdno(rpmfile)
    epoch = rpmhdr[rpm.RPMTAG_EPOCHNUM]
    name = to_native(rpmhdr[rpm.RPMTAG_NAME])
    version = to_native(rpmhdr[rpm.RPMTAG_VERSION])
    release = to_native(rpmhdr[rpm.RPMTAG_RELEASE])
    arch = to_native(rpmhdr[rpm.RPMTAG_ARCH])
    if arch == 'noarch' and rpmhdr[rpm.RPMTAG_SOURCEPACKAGE] == 1:
        arch = 'src'
    return (name, epoch, version, release, arch)