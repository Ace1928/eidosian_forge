from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def zapi_apply(self, current):
    changed = False
    if self.parameters['state'] == 'present':
        if current:
            if self.parameters['status'] == 'up':
                if not self.current_status():
                    if not self.module.check_mode:
                        self.start_fcp()
                    changed = True
            elif self.current_status():
                if not self.module.check_mode:
                    self.stop_fcp()
                changed = True
        else:
            if not self.module.check_mode:
                self.create_fcp()
                if self.parameters['status'] == 'up':
                    self.start_fcp()
                elif self.parameters['status'] == 'down':
                    self.stop_fcp()
            changed = True
    elif current:
        if not self.module.check_mode:
            if self.current_status():
                self.stop_fcp()
            self.destroy_fcp()
        changed = True
    return changed