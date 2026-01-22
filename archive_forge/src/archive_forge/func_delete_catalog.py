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
def delete_catalog(module, rest_obj, catalog_list):
    delete_ids = [d['Id'] for d in catalog_list]
    validate_delete_operation(rest_obj, module, catalog_list, delete_ids)
    delete_payload = {'CatalogIds': delete_ids}
    rest_obj.invoke_request('POST', DELETE_CATALOG_URI, data=delete_payload)
    module.exit_json(msg=CATALOG_DEL_SUCCESS, changed=True, catalog_id=delete_ids)