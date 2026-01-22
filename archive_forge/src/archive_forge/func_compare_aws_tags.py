from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.six import string_types
def compare_aws_tags(current_tags_dict, new_tags_dict, purge_tags=True):
    """
    Compare two dicts of AWS tags. Dicts are expected to of been created using 'boto3_tag_list_to_ansible_dict' helper function.
    Two dicts are returned - the first is tags to be set, the second is any tags to remove. Since the AWS APIs differ
    these may not be able to be used out of the box.

    :param current_tags_dict:
    :param new_tags_dict:
    :param purge_tags:
    :return: tag_key_value_pairs_to_set: a dict of key value pairs that need to be set in AWS. If all tags are identical this dict will be empty
    :return: tag_keys_to_unset: a list of key names (type str) that need to be unset in AWS. If no tags need to be unset this list will be empty
    """
    tag_key_value_pairs_to_set = {}
    tag_keys_to_unset = []
    if purge_tags:
        for key in current_tags_dict.keys():
            if key in new_tags_dict:
                continue
            if key.startswith('aws:'):
                continue
            tag_keys_to_unset.append(key)
    for key in set(new_tags_dict.keys()) - set(tag_keys_to_unset):
        if to_text(new_tags_dict[key]) != current_tags_dict.get(key):
            tag_key_value_pairs_to_set[key] = new_tags_dict[key]
    return (tag_key_value_pairs_to_set, tag_keys_to_unset)