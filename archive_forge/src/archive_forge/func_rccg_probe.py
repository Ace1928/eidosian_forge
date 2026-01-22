from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def rccg_probe(self, data):
    props = {}
    propscv = {}
    if self.copytype and self.copytype != data['copy_type']:
        if self.copytype == 'global':
            props['global'] = True
        elif self.copytype == 'metro':
            props['metro'] = True
        else:
            self.module.fail_json(msg="Unsupported mirror type: %s. Only 'global' and 'metro' are supported when modifying" % self.copytype)
    if self.copytype == 'global' and self.cyclingperiod and (self.cyclingperiod != int(data['cycle_period_seconds'])):
        propscv['cycleperiodseconds'] = self.cyclingperiod
    if self.copytype == 'global' and self.cyclingmode and (self.cyclingmode != data['cycling_mode']):
        propscv['cyclingmode'] = self.cyclingmode
    return (props, propscv)