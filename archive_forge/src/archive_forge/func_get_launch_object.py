import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def get_launch_object(connection, ec2_connection):
    launch_object = dict()
    launch_config_name = module.params.get('launch_config_name')
    launch_template = module.params.get('launch_template')
    mixed_instances_policy = module.params.get('mixed_instances_policy')
    if launch_config_name is None and launch_template is None:
        return launch_object
    elif launch_config_name:
        try:
            launch_configs = describe_launch_configurations(connection, launch_config_name)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to describe launch configurations')
        if len(launch_configs['LaunchConfigurations']) == 0:
            module.fail_json(msg=f'No launch config found with name {launch_config_name}')
        launch_object = {'LaunchConfigurationName': launch_configs['LaunchConfigurations'][0]['LaunchConfigurationName']}
        return launch_object
    elif launch_template:
        lt = describe_launch_templates(ec2_connection, launch_template)['LaunchTemplates'][0]
        if launch_template['version'] is not None:
            launch_object = {'LaunchTemplate': {'LaunchTemplateId': lt['LaunchTemplateId'], 'Version': launch_template['version']}}
        else:
            launch_object = {'LaunchTemplate': {'LaunchTemplateId': lt['LaunchTemplateId'], 'Version': str(lt['LatestVersionNumber'])}}
        if mixed_instances_policy:
            instance_types = mixed_instances_policy.get('instance_types', [])
            instances_distribution = mixed_instances_policy.get('instances_distribution', {})
            policy = {'LaunchTemplate': {'LaunchTemplateSpecification': launch_object['LaunchTemplate']}}
            if instance_types:
                policy['LaunchTemplate']['Overrides'] = []
                for instance_type in instance_types:
                    instance_type_dict = {'InstanceType': instance_type}
                    policy['LaunchTemplate']['Overrides'].append(instance_type_dict)
            if instances_distribution:
                instances_distribution_params = scrub_none_parameters(instances_distribution)
                policy['InstancesDistribution'] = snake_dict_to_camel_dict(instances_distribution_params, capitalize_first=True)
            launch_object['MixedInstancesPolicy'] = policy
        return launch_object