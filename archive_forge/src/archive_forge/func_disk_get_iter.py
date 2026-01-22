from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def disk_get_iter(self, name):
    """
        Return storage-disk-get-iter query results
        Filter disk list by aggregate name, and only reports disk-name and plex-name
        :param name: Name of the aggregate
        :return: NaElement
        """
    disk_get_iter = netapp_utils.zapi.NaElement('storage-disk-get-iter')
    query_details = {'query': {'storage-disk-info': {'disk-raid-info': {'disk-aggregate-info': {'aggregate-name': name}}}}}
    disk_get_iter.translate_struct(query_details)
    attributes = {'desired-attributes': {'storage-disk-info': {'disk-name': None, 'disk-raid-info': {'disk_aggregate_info': {'plex-name': None}}}}}
    disk_get_iter.translate_struct(attributes)
    result = None
    try:
        result = self.server.invoke_successfully(disk_get_iter, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting disks: %s' % to_native(error), exception=traceback.format_exc())
    return result