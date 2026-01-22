import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.core import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def wait_for_instance_snapshot(module, client, instance_snapshot_name):
    wait_timeout = module.params.get('wait_timeout')
    wait_max = time.time() + wait_timeout
    snapshot = find_instance_snapshot_info(module, client, instance_snapshot_name)
    while wait_max > time.time():
        snapshot = find_instance_snapshot_info(module, client, instance_snapshot_name)
        current_state = snapshot['state']
        if current_state != 'pending':
            break
        time.sleep(5)
    else:
        module.fail_json(msg=f'Timed out waiting for instance snapshot "{instance_snapshot_name}" to be created.')
    return snapshot