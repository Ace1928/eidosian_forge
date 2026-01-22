from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def upload_release_catalog(self, content):
    self.vsanVumSystem.VsanVcUploadReleaseDb(db=content)