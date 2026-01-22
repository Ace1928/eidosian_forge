from functools import cmp_to_key
import ansible.module_utils.common.warnings as ansible_warnings
from ansible.module_utils._text import to_text
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import string_types
def compare_policies(current_policy, new_policy, default_version='2008-10-17'):
    """Compares the existing policy and the updated policy
    Returns True if there is a difference between policies.
    """
    if default_version:
        if isinstance(current_policy, dict):
            current_policy = current_policy.copy()
            current_policy.setdefault('Version', default_version)
        if isinstance(new_policy, dict):
            new_policy = new_policy.copy()
            new_policy.setdefault('Version', default_version)
    return set(_hashable_policy(new_policy, [])) != set(_hashable_policy(current_policy, []))