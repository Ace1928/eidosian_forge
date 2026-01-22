import re
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_launch_configs(client, module):
    name_regex = module.params.get('name_regex')
    sort_order = module.params.get('sort_order')
    limit = module.params.get('limit')
    paginator = client.get_paginator('describe_launch_configurations')
    response_iterator = paginator.paginate(PaginationConfig={'MaxItems': 1000, 'PageSize': 100})
    results = []
    for response in response_iterator:
        response['LaunchConfigurations'] = filter(lambda lc: re.compile(name_regex).match(lc['LaunchConfigurationName']), response['LaunchConfigurations'])
        for lc in response['LaunchConfigurations']:
            data = {'name': lc['LaunchConfigurationName'], 'arn': lc['LaunchConfigurationARN'], 'created_time': lc['CreatedTime'], 'user_data': lc['UserData'], 'instance_type': lc['InstanceType'], 'image_id': lc['ImageId'], 'ebs_optimized': lc['EbsOptimized'], 'instance_monitoring': lc['InstanceMonitoring'], 'classic_link_vpc_security_groups': lc['ClassicLinkVPCSecurityGroups'], 'block_device_mappings': lc['BlockDeviceMappings'], 'keyname': lc['KeyName'], 'security_groups': lc['SecurityGroups'], 'kernel_id': lc['KernelId'], 'ram_disk_id': lc['RamdiskId'], 'associate_public_address': lc.get('AssociatePublicIpAddress', False)}
            results.append(data)
    results.sort(key=lambda e: e['name'], reverse=sort_order == 'descending')
    if limit:
        results = results[:int(limit)]
    module.exit_json(changed=False, results=results)