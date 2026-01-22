from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def check_changed_parameter_values(values, old_parameters, new_parameters):
    """Checking if the new values are different than the old values."""
    changed_with_update = False
    if values:
        for parameter in values:
            if old_parameters[parameter] != new_parameters[parameter]:
                changed_with_update = True
                break
    else:
        for parameter in old_parameters:
            if old_parameters[parameter] != new_parameters[parameter]:
                changed_with_update = True
                break
    return changed_with_update