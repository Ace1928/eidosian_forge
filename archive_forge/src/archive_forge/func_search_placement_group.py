from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.exponential_backoff()
def search_placement_group(connection, module):
    """
    Check if a placement group exists.
    """
    name = module.params.get('name')
    try:
        response = connection.describe_placement_groups(Filters=[{'Name': 'group-name', 'Values': [name]}])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f"Couldn't find placement group named [{name}]")
    if len(response['PlacementGroups']) != 1:
        return None
    else:
        placement_group = response['PlacementGroups'][0]
        return {'name': placement_group['GroupName'], 'state': placement_group['State'], 'strategy': placement_group['Strategy']}