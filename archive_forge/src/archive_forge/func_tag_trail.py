from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def tag_trail(module, client, tags, trail_arn, curr_tags=None, purge_tags=True):
    """
    Creates, updates, removes tags on a CloudTrail resource

    module : AnsibleAWSModule object
    client : boto3 client connection object
    tags : Dict of tags converted from ansible_dict to boto3 list of dicts
    trail_arn : The ARN of the CloudTrail to operate on
    curr_tags : Dict of the current tags on resource, if any
    dry_run : true/false to determine if changes will be made if needed
    """
    if tags is None:
        return False
    curr_tags = curr_tags or {}
    tags_to_add, tags_to_remove = compare_aws_tags(curr_tags, tags, purge_tags=purge_tags)
    if not tags_to_add and (not tags_to_remove):
        return False
    if module.check_mode:
        return True
    if tags_to_remove:
        remove = {k: curr_tags[k] for k in tags_to_remove}
        tags_to_remove = ansible_dict_to_boto3_tag_list(remove)
        try:
            client.remove_tags(ResourceId=trail_arn, TagsList=tags_to_remove)
        except (BotoCoreError, ClientError) as err:
            module.fail_json_aws(err, msg='Failed to remove tags from Trail')
    if tags_to_add:
        tags_to_add = ansible_dict_to_boto3_tag_list(tags_to_add)
        try:
            client.add_tags(ResourceId=trail_arn, TagsList=tags_to_add)
        except (BotoCoreError, ClientError) as err:
            module.fail_json_aws(err, msg='Failed to add tags to Trail')
    return True