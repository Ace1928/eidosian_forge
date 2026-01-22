from __future__ import absolute_import, division, print_function
import stat
import os
import traceback
from ansible.module_utils.common import respawn
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
@property
def short_chroot(self):
    """str: Chroot (distribution-version-architecture) shorten to distribution-version."""
    return self.chroot.rsplit('-', 1)[0]