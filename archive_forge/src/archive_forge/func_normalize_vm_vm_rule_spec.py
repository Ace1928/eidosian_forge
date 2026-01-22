from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
@staticmethod
def normalize_vm_vm_rule_spec(rule_obj=None):
    """
        Return human readable rule spec
        Args:
            rule_obj: Rule managed object

        Returns: Dictionary with DRS VM VM Rule info

        """
    if rule_obj is None:
        return {}
    return dict(rule_key=rule_obj.key, rule_enabled=rule_obj.enabled, rule_name=rule_obj.name, rule_mandatory=rule_obj.mandatory, rule_uuid=rule_obj.ruleUuid, rule_vms=[vm.name for vm in rule_obj.vm], rule_type='vm_vm_rule', rule_affinity=True if isinstance(rule_obj, vim.cluster.AffinityRuleSpec) else False)