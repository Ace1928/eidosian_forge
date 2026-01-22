from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def list_layer_versions(lambda_client, name):
    try:
        layer_versions = _list_layer_versions(lambda_client, LayerName=name)['LayerVersions']
        return [camel_dict_to_snake_dict(layer) for layer in layer_versions]
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        raise LambdaLayerFailure(e, f'Unable to list layer versions for name {name}')