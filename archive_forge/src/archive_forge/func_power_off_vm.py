from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def power_off_vm(self):
    self.log('Powered off virtual machine {0} - Skip_Shutdown {1}'.format(self.name, self.force))
    self.results['actions'].append('Powered off virtual machine {0} - Skip_Shutdown {1}'.format(self.name, self.force))
    try:
        poller = self.compute_client.virtual_machines.begin_power_off(self.resource_group, self.name, skip_shutdown=self.force)
        self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error powering off virtual machine {0} - {1}'.format(self.name, str(exc)))
    return True