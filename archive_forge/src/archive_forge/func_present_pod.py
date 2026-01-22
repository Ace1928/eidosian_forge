from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_pod(self):
    pod = self.get_pod()
    if pod:
        pod = self._update_pod()
    else:
        pod = self._create_pod()
    return pod