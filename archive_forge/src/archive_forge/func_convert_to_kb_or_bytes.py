from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def convert_to_kb_or_bytes(self, option):
    """
        convert input to kb, and set to self.parameters.
        :param option: disk_limit or soft_disk_limit.
        :return: boolean if it can be converted.
        """
    self.parameters[option].replace(' ', '')
    slices = re.findall('\\d+|\\D+', self.parameters[option])
    if len(slices) < 1 or len(slices) > 2:
        return False
    if not slices[0].isdigit():
        return False
    if len(slices) > 1 and slices[1].lower() not in ['b', 'kb', 'mb', 'gb', 'tb']:
        return False
    if len(slices) == 1 and self.use_rest:
        slices = (slices[0], 'kb')
    if len(slices) > 1:
        if not self.use_rest:
            self.parameters[option] = str(int(slices[0]) * netapp_utils.POW2_BYTE_MAP[slices[1].lower()] // 1024)
        else:
            self.parameters[option] = str(int(slices[0]) * netapp_utils.POW2_BYTE_MAP[slices[1].lower()])
    if self.use_rest:
        self.parameters[option] = str((int(self.parameters[option]) + 1023) // 1024 * 1024)
    return True