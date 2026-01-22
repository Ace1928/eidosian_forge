from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
@staticmethod
def normalize_rule_spec(rule_obj=None):
    """
        Return human readable rule spec
        Args:
            rule_obj: Rule managed object

        Returns: Dictionary with Rule info

        """
    if rule_obj is None:
        return {}
    return dict(rule_key=rule_obj.key, rule_enabled=rule_obj.enabled, rule_name=rule_obj.name, rule_mandatory=rule_obj.mandatory, rule_uuid=rule_obj.ruleUuid, rule_vms=[vm.name for vm in rule_obj.vm], rule_affinity=True if isinstance(rule_obj, vim.cluster.AffinityRuleSpec) else False)