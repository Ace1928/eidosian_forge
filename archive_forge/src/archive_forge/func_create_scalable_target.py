from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_scalable_target(connection, module):
    changed = False
    try:
        scalable_targets = connection.describe_scalable_targets(ServiceNamespace=module.params.get('service_namespace'), ResourceIds=[module.params.get('resource_id')], ScalableDimension=module.params.get('scalable_dimension'))
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe scalable targets')
    if not scalable_targets['ScalableTargets'] or (module.params.get('override_task_capacity') and (scalable_targets['ScalableTargets'][0]['MinCapacity'] != module.params.get('minimum_tasks') or scalable_targets['ScalableTargets'][0]['MaxCapacity'] != module.params.get('maximum_tasks'))):
        changed = True
        try:
            connection.register_scalable_target(ServiceNamespace=module.params.get('service_namespace'), ResourceId=module.params.get('resource_id'), ScalableDimension=module.params.get('scalable_dimension'), MinCapacity=module.params.get('minimum_tasks'), MaxCapacity=module.params.get('maximum_tasks'))
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to register scalable target')
    try:
        response = connection.describe_scalable_targets(ServiceNamespace=module.params.get('service_namespace'), ResourceIds=[module.params.get('resource_id')], ScalableDimension=module.params.get('scalable_dimension'))
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe scalable targets')
    if response['ScalableTargets']:
        snaked_response = camel_dict_to_snake_dict(response['ScalableTargets'][0])
    else:
        snaked_response = {}
    return {'changed': changed, 'response': snaked_response}