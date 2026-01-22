from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def delete_broker(conn, module, broker_id):
    wait = module.params.get('wait')
    try:
        response = conn.delete_broker(BrokerId=broker_id)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't delete broker.")
    if wait:
        wait_for_status(conn, module)
    return response