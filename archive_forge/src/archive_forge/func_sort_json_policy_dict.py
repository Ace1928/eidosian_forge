from functools import cmp_to_key
import ansible.module_utils.common.warnings as ansible_warnings
from ansible.module_utils._text import to_text
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import string_types
def sort_json_policy_dict(policy_dict):
    """
    DEPRECATED - will be removed in amazon.aws 8.0.0

    Sort any lists in an IAM JSON policy so that comparison of two policies with identical values but
    different orders will return true
    Args:
        policy_dict (dict): Dict representing IAM JSON policy.
    Basic Usage:
        >>> my_iam_policy = {'Principle': {'AWS':["31","7","14","101"]}
        >>> sort_json_policy_dict(my_iam_policy)
    Returns:
        Dict: Will return a copy of the policy as a Dict but any List will be sorted
        {
            'Principle': {
                'AWS': [ '7', '14', '31', '101' ]
            }
        }
    """
    ansible_warnings.deprecate('amazon.aws.module_utils.policy.sort_json_policy_dict has been deprecated, consider using amazon.aws.module_utils.policy.compare_policies instead', version='8.0.0', collection_name='amazon.aws')

    def value_is_list(my_list):
        checked_list = []
        for item in my_list:
            if isinstance(item, dict):
                checked_list.append(sort_json_policy_dict(item))
            elif isinstance(item, list):
                checked_list.append(value_is_list(item))
            else:
                checked_list.append(item)
        checked_list.sort(key=lambda x: sorted(x.items()) if isinstance(x, dict) else x)
        return checked_list
    ordered_policy_dict = {}
    for key, value in policy_dict.items():
        if isinstance(value, dict):
            ordered_policy_dict[key] = sort_json_policy_dict(value)
        elif isinstance(value, list):
            ordered_policy_dict[key] = value_is_list(value)
        else:
            ordered_policy_dict[key] = value
    return ordered_policy_dict