from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def get_latest_engine_version(conn, module, engine_type):
    try:
        response = conn.describe_broker_engine_types(EngineType=engine_type)
        return response['BrokerEngineTypes'][0]['EngineVersions'][0]['Name']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't list engine versions")