from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def subscribe_pool(self, regexp):
    """
            Subscribe current system to available pools matching the specified
            regular expression
            Raises:
              * Exception - if error occurs while running command
        """
    available_pools = RhsmPools(self.module)
    subscribed_pool_ids = []
    for pool in available_pools.filter_pools(regexp):
        pool.subscribe()
        subscribed_pool_ids.append(pool.get_pool_id())
    return subscribed_pool_ids