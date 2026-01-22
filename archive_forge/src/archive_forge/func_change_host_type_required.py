from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_host_type_required(self):
    """Determine whether default host type change is required."""
    if self.host_type_index is None:
        return False
    current_configuration = self.get_current_configuration()
    current_available_host_types = current_configuration['host_type_options']
    if isinstance(self.host_type_index, str):
        self.host_type_index = self.host_type_index.lower()
    if self.host_type_index in self.HOST_TYPE_INDEXES.keys():
        self.host_type_index = self.HOST_TYPE_INDEXES[self.host_type_index]
    elif self.host_type_index in current_available_host_types.keys():
        self.host_type_index = current_available_host_types[self.host_type_index]
    if self.host_type_index not in current_available_host_types.values():
        self.module.fail_json(msg='Invalid host type index! Array [%s]. Available host options [%s].' % (self.ssid, current_available_host_types))
    return int(self.host_type_index) != current_configuration['default_host_type_index']