import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def elb_dreg(asg_connection, group_name, instance_id):
    as_group = describe_autoscaling_groups(asg_connection, group_name)[0]
    wait_timeout = module.params.get('wait_timeout')
    count = 1
    if as_group['LoadBalancerNames'] and as_group['HealthCheckType'] == 'ELB':
        elb_connection = module.client('elb')
    else:
        return
    for lb in as_group['LoadBalancerNames']:
        deregister_lb_instances(elb_connection, lb, instance_id)
        module.debug(f'De-registering {instance_id} from ELB {lb}')
    wait_timeout = time.time() + wait_timeout
    while wait_timeout > time.time() and count > 0:
        count = 0
        for lb in as_group['LoadBalancerNames']:
            lb_instances = describe_instance_health(elb_connection, lb, [])
            for i in lb_instances['InstanceStates']:
                if i['InstanceId'] == instance_id and i['State'] == 'InService':
                    count += 1
                    module.debug(f'{i['InstanceId']}: {i['State']}, {i['Description']}')
        time.sleep(10)
    if wait_timeout <= time.time():
        module.fail_json(msg=f'Waited too long for instance to deregister. {time.asctime()}')