from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_trail_detail(connection, module):
    output = {}
    trail_name_list = module.params.get('trail_names')
    include_shadow_trails = module.params.get('include_shadow_trails')
    if not trail_name_list:
        trail_name_list = get_trails(connection, module)
    try:
        result = connection.describe_trails(trailNameList=trail_name_list, includeShadowTrails=include_shadow_trails, aws_retry=True)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to get the trails.')
    snaked_cloud_trail = []
    for cloud_trail in result['trailList']:
        try:
            status_dict = connection.get_trail_status(Name=cloud_trail['TrailARN'], aws_retry=True)
            cloud_trail.update(status_dict)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to get the trail status')
        try:
            tag_list = connection.list_tags(ResourceIdList=[cloud_trail['TrailARN']])
            for tag_dict in tag_list['ResourceTagList']:
                cloud_trail.update(tag_dict)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.warn(f'Failed to get the trail tags - {e}')
        snaked_cloud_trail.append(camel_dict_to_snake_dict(cloud_trail))
    for tr in snaked_cloud_trail:
        if 'tags_list' in tr:
            tr['tags'] = boto3_tag_list_to_ansible_dict(tr['tags_list'], 'key', 'value')
            del tr['tags_list']
        if 'response_metadata' in tr:
            del tr['response_metadata']
    output['trail_list'] = snaked_cloud_trail
    return output