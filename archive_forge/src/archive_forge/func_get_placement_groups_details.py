from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_placement_groups_details(connection, module):
    names = module.params.get('names')
    try:
        if len(names) > 0:
            response = connection.describe_placement_groups(Filters=[{'Name': 'group-name', 'Values': names}])
        else:
            response = connection.describe_placement_groups()
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f"Couldn't find placement groups named [{names}]")
    results = []
    for placement_group in response['PlacementGroups']:
        results.append({'name': placement_group['GroupName'], 'state': placement_group['State'], 'strategy': placement_group['Strategy']})
    return results