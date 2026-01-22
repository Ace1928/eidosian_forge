from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def fcconsistgrp_probe(self, data):
    props = {}
    self.log('Probe which properties need to be updated...')
    if not self.noownershipgroup:
        if self.ownershipgroup and self.ownershipgroup != data['owner_name']:
            props['ownershipgroup'] = self.ownershipgroup
    if self.noownershipgroup and data['owner_name']:
        props['noownershipgroup'] = self.noownershipgroup
    return props