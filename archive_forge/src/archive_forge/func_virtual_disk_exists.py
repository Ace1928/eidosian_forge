from __future__ import absolute_import, division, print_function
import time
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def virtual_disk_exists(self):
    """Checks if a virtual disk exists for a guest

        The virtual disk names can differ based on the device vCMP is installed on.
        For instance, on a shuttle-series device with no slots, you will see disks
        that resemble the following

          guest1.img

        On an 8-blade Viprion with slots though, you will see

          guest1.img/1

        The "/1" in this case is the slot that it is a part of. This method looks
        for the virtual-disk without the trailing slot.

        Returns:
            dict
        """
    response = self.get_virtual_disks_on_device()
    check = '{0}'.format(self.have.virtual_disk)
    for resource in response['items']:
        if resource['name'].startswith(check):
            return True
        else:
            return False