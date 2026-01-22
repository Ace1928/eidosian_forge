from __future__ import absolute_import, division, print_function
import shutil
import traceback
from os import path
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
def init_layman(config=None):
    """Returns the initialized ``LaymanAPI``.

    :param config: the layman's configuration to use (optional)
    """
    if config is None:
        config = BareConfig(read_configfile=True, quietness=1)
    return LaymanAPI(config)