from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def server_type_and_devices(self):
    """Compares difference between server type and devices list

        These two parameters are linked with each other and, therefore, must be
        compared together to ensure that the correct setting is sent to BIG-IP

        :return:
        """
    devices_change = self._devices_changed()
    server_change = self._server_type_changed()
    if not devices_change and (not server_change):
        return None
    tmos = tmos_version(self.client)
    if Version(tmos) >= Version('13.0.0'):
        result = self._handle_current_server_type_and_devices(devices_change, server_change)
        return result
    else:
        result = self._handle_legacy_server_type_and_devices(devices_change, server_change)
        return result