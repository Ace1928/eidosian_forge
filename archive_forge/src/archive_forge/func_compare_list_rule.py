from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def compare_list_rule(old_rule, rule, key):
    return set(map(str, rule.get(key) or [])) != set(map(str, old_rule.get(key) or []))