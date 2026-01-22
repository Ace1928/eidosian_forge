from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def replication_policy_probe(self):
    field_mappings = (('topology', self.rp_data.get('topology', '')), ('location1system', (('location1_system_name', self.rp_data.get('location1_system_name', '')), ('location1_system_id', self.rp_data.get('location1_system_id', '')))), ('location1iogrp', self.rp_data.get('location1_iogrp_id', '')), ('location2system', (('location2_system_name', self.rp_data.get('location2_system_name', '')), ('location2_system_id', self.rp_data.get('location2_system_id', '')))), ('location2iogrp', self.rp_data.get('location2_iogrp_id', '')), ('rpoalert', self.rp_data.get('rpo_alert', '')))
    self.log('replication policy probe data: %s', field_mappings)
    for f, v in field_mappings:
        current_value = str(getattr(self, f))
        if current_value and f in {'location1system', 'location2system'}:
            try:
                next(iter(filter(lambda val: val[1] == current_value, v)))
            except StopIteration:
                self.module.fail_json(msg='Policy modification is not supported. Please delete and recreate new policy.')
        elif current_value and current_value != v:
            self.module.fail_json(msg='Policy modification is not supported. Please delete and recreate new policy.')