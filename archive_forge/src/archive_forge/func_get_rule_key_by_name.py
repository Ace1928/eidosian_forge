from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_rule_key_by_name(self, cluster_obj=None, rule_name=None):
    """
        Get a specific DRS rule key by name
        Args:
            rule_name: Name of rule
            cluster_obj: Cluster managed object

        Returns: Rule Object if found or None

        """
    if cluster_obj is None:
        cluster_obj = self.cluster_obj
    if rule_name:
        rules_list = [rule for rule in cluster_obj.configuration.rule if rule.name == rule_name]
        if rules_list:
            return rules_list[0]
    return None