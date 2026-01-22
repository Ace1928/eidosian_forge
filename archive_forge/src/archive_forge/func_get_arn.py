from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_arn(ecs, module, cluster_name, resource_type, resource):
    try:
        if resource_type == 'cluster':
            description = ecs.describe_clusters(clusters=[resource])
            resource_arn = description['clusters'][0]['clusterArn']
        elif resource_type == 'task':
            description = ecs.describe_tasks(cluster=cluster_name, tasks=[resource])
            resource_arn = description['tasks'][0]['taskArn']
        elif resource_type == 'service':
            description = ecs.describe_services(cluster=cluster_name, services=[resource])
            resource_arn = description['services'][0]['serviceArn']
        elif resource_type == 'task_definition':
            description = ecs.describe_task_definition(taskDefinition=resource)
            resource_arn = description['taskDefinition']['taskDefinitionArn']
        elif resource_type == 'container':
            description = ecs.describe_container_instances(clusters=[resource])
            resource_arn = description['containerInstances'][0]['containerInstanceArn']
    except (IndexError, KeyError):
        module.fail_json(msg=f'Failed to find {resource_type} {resource}')
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed to find {resource_type} {resource}')
    return resource_arn