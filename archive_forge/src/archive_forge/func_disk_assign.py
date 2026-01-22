from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def disk_assign(self, needed_disks):
    """
        Assign disks to node
        """
    if self.use_rest:
        api = 'private/cli/storage/disk/assign'
        if needed_disks > 0:
            body = {'owner': self.parameters['node'], 'count': needed_disks}
            if 'disk_type' in self.parameters:
                body['type'] = self.parameters['disk_type']
        else:
            body = {'node': self.parameters['node'], 'all': True}
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        if needed_disks > 0:
            assign_disk = netapp_utils.zapi.NaElement.create_node_with_children('disk-sanown-assign', **{'owner': self.parameters['node'], 'disk-count': str(needed_disks)})
            if 'disk_type' in self.parameters:
                assign_disk.add_new_child('disk-type', self.parameters['disk_type'])
        else:
            assign_disk = netapp_utils.zapi.NaElement.create_node_with_children('disk-sanown-assign', **{'node-name': self.parameters['node'], 'all': 'true'})
        try:
            self.server.invoke_successfully(assign_disk, enable_tunneling=True)
            return True
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error assigning disks %s' % to_native(error), exception=traceback.format_exc())