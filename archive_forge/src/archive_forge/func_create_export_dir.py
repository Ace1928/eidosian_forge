from __future__ import absolute_import, division, print_function
import os
import hashlib
from time import sleep
from threading import Thread
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def create_export_dir(self, vm_obj):
    self.ovf_dir = os.path.join(self.params['export_dir'], vm_obj.name)
    if not os.path.exists(self.ovf_dir):
        try:
            os.makedirs(self.ovf_dir)
        except OSError as err:
            self.module.fail_json(msg='Exception caught when create folder %s, with error %s' % (self.ovf_dir, to_text(err)))
    self.mf_file = os.path.join(self.ovf_dir, vm_obj.name + '.mf')