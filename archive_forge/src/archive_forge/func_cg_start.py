from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cg_start(self):
    """
        For the given list of volumes, creates cg-snapshot
        """
    snapshot_started = False
    cgstart = netapp_utils.zapi.NaElement('cg-start')
    cgstart.add_new_child('snapshot', self.parameters['snapshot'])
    cgstart.add_new_child('timeout', self.parameters['timeout'])
    volume_list = netapp_utils.zapi.NaElement('volumes')
    cgstart.add_child_elem(volume_list)
    for vol in self.parameters['volumes']:
        snapshot_exists = self.does_snapshot_exist(vol)
        if snapshot_exists is None:
            snapshot_started = True
            volume_list.add_new_child('volume-name', vol)
    if snapshot_started:
        if self.parameters.get('snapmirror_label') is not None:
            cgstart.add_new_child('snapmirror-label', self.parameters['snapmirror_label'])
        try:
            cgresult = self.server.invoke_successfully(cgstart, enable_tunneling=True)
            if cgresult.get_child_by_name('cg-id'):
                self.cgid = cgresult['cg-id']
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating CG snapshot %s: %s' % (self.parameters['snapshot'], to_native(error)), exception=traceback.format_exc())
    return snapshot_started