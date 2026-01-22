from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.six import string_types
def boto3_tag_list_to_ansible_dict(tags_list, tag_name_key_name=None, tag_value_key_name=None):
    """Convert a boto3 list of resource tags to a flat dict of key:value pairs
    Args:
        tags_list (list): List of dicts representing AWS tags.
        tag_name_key_name (str): Value to use as the key for all tag keys (useful because boto3 doesn't always use "Key")
        tag_value_key_name (str): Value to use as the key for all tag values (useful because boto3 doesn't always use "Value")
    Basic Usage:
        >>> tags_list = [{'Key': 'MyTagKey', 'Value': 'MyTagValue'}]
        >>> boto3_tag_list_to_ansible_dict(tags_list)
        [
            {
                'Key': 'MyTagKey',
                'Value': 'MyTagValue'
            }
        ]
    Returns:
        Dict: Dict of key:value pairs representing AWS tags
         {
            'MyTagKey': 'MyTagValue',
        }
    """
    if tag_name_key_name and tag_value_key_name:
        tag_candidates = {tag_name_key_name: tag_value_key_name}
    else:
        tag_candidates = {'key': 'value', 'Key': 'Value'}
    if not tags_list or not any((tag for tag in tags_list)):
        return {}
    for k, v in tag_candidates.items():
        if k in tags_list[0] and v in tags_list[0]:
            return dict(((tag[k], tag[v]) for tag in tags_list))
    raise ValueError(f"Couldn't find tag key (candidates {str(tag_candidates)}) in tag list {str(tags_list)}")