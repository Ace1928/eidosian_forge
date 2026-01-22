import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def wait_for_new_inst(connection, group_name, wait_timeout, desired_size, prop):
    as_group = describe_autoscaling_groups(connection, group_name)[0]
    props = get_properties(as_group)
    module.debug(f'Waiting for {prop} = {desired_size}, currently {props[prop]}')
    wait_timeout = time.time() + wait_timeout
    while wait_timeout > time.time() and desired_size > props[prop]:
        module.debug(f'Waiting for {prop} = {desired_size}, currently {props[prop]}')
        time.sleep(10)
        as_group = describe_autoscaling_groups(connection, group_name)[0]
        props = get_properties(as_group)
    if wait_timeout <= time.time():
        module.fail_json(msg=f'Waited too long for new instances to become viable. {time.asctime()}')
    module.debug(f'Reached {prop}: {desired_size}')
    return props