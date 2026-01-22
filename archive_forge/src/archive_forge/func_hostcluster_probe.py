from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def hostcluster_probe(self, data):
    props = []
    if self.removeallhosts:
        self.module.fail_json(msg="Parameter 'removeallhosts' can be used only while deleting hostcluster")
    if self.ownershipgroup and self.noownershipgroup:
        self.module.fail_json(msg="You must not pass in both 'ownershipgroup' and 'noownershipgroup' to the module.")
    if data['owner_name'] and self.noownershipgroup:
        props += ['noownershipgroup']
    if self.ownershipgroup and (not data['owner_name'] or self.ownershipgroup != data['owner_name']):
        props += ['ownershipgroup']
    if props is []:
        props = None
    self.log("hostcluster_probe props='%s'", data)
    return props