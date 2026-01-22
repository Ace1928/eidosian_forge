from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def validate_dell_online(all_catalog, module):
    """
    only one dell_online repository type catalog creation is possible from ome
    """
    catalog_name = module.params['catalog_name'][0]
    for name, repo_type in all_catalog.items():
        if repo_type == 'DELL_ONLINE' and name != catalog_name:
            module.fail_json(msg=DELL_ONLINE_EXISTS.format(catalog_name=name))