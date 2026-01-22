from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.service.service import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.service import (
def get_service_data(self, connection):
    return connection.get('show running-config all | section ^service ')