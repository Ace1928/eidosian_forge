from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def snapmirror_get_iter(self, destination=None):
    """
        Compose NaElement object to query current SnapMirror relations using destination-path
        SnapMirror relation for a destination path is unique
        :return: NaElement object for SnapMirror-get-iter
        """
    snapmirror_get_iter = netapp_utils.zapi.NaElement('snapmirror-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    snapmirror_info = netapp_utils.zapi.NaElement('snapmirror-info')
    if destination is None:
        destination = self.parameters['destination_path']
    snapmirror_info.add_new_child('destination-location', destination)
    query.add_child_elem(snapmirror_info)
    snapmirror_get_iter.add_child_elem(query)
    return snapmirror_get_iter