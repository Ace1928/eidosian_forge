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
@staticmethod
def chroot_conf():
    """Obtain information about the distribution, version, and architecture of the target.

        Returns:
            Chroot info in the form of distribution-version-architecture.
        """
    distribution, version, codename = distro.linux_distribution(full_distribution_name=False)
    base = CoprModule.get_base()
    return '{0}-{1}-{2}'.format(distribution, version, base.conf.arch)