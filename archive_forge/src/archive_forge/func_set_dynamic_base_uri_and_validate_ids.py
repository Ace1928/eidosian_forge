from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def set_dynamic_base_uri_and_validate_ids(self):
    network_device_function_id_uri = self.__perform_validation_for_network_device_function_id()
    resp = get_dynamic_uri(self.idrac, network_device_function_id_uri)
    self.oem_uri = resp.get('Links', {}).get('Oem', {}).get('Dell', {}).get('DellNetworkAttributes', {}).get('@odata.id', {})
    self.redfish_uri = network_device_function_id_uri