from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def list_load_balancers(connection, module):
    load_balancer_arns = module.params.get('load_balancer_arns')
    names = module.params.get('names')
    include_attributes = module.params.get('include_attributes')
    include_listeners = module.params.get('include_listeners')
    include_listener_rules = module.params.get('include_listener_rules')
    try:
        if not load_balancer_arns and (not names):
            load_balancers = get_paginator(connection)
        if load_balancer_arns:
            load_balancers = get_paginator(connection, LoadBalancerArns=load_balancer_arns)
        if names:
            load_balancers = get_paginator(connection, Names=names)
    except is_boto3_error_code('LoadBalancerNotFound'):
        module.exit_json(load_balancers=[])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to list load balancers')
    for load_balancer in load_balancers['LoadBalancers']:
        if include_attributes:
            load_balancer.update(get_load_balancer_attributes(connection, module, load_balancer['LoadBalancerArn']))
        if include_listeners or include_listener_rules:
            load_balancer['listeners'] = get_alb_listeners(connection, module, load_balancer['LoadBalancerArn'])
        if include_listener_rules:
            for listener in load_balancer['listeners']:
                listener['rules'] = get_listener_rules(connection, module, listener['ListenerArn'])
    snaked_load_balancers = [camel_dict_to_snake_dict(load_balancer) for load_balancer in load_balancers['LoadBalancers']]
    for snaked_load_balancer in snaked_load_balancers:
        snaked_load_balancer['tags'] = get_load_balancer_tags(connection, module, snaked_load_balancer['load_balancer_arn'])
    module.exit_json(load_balancers=snaked_load_balancers)