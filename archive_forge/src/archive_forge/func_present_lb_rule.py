from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_lb_rule(self):
    required_params = ['algorithm', 'private_port', 'public_port']
    self.module.fail_on_missing_params(required_params=required_params)
    args = self._get_common_args()
    rule = self.get_rule(**args)
    if rule:
        rule = self._update_lb_rule(rule)
    else:
        rule = self._create_lb_rule(rule)
    if rule:
        rule = self.ensure_tags(resource=rule, resource_type='LoadBalancer')
    return rule