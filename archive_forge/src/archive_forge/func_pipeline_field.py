import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def pipeline_field(client, dp_id, field):
    """Return a pipeline field from the pipeline description.

    The available fields are listed in describe_pipelines output.

    :param object client: boto3 datapipeline client
    :param string dp_id: pipeline id
    :param string field: pipeline description field
    :returns: pipeline field information

    """
    dp_description = pipeline_description(client, dp_id)
    for field_key in dp_description['pipelineDescriptionList'][0]['fields']:
        if field_key['key'] == field:
            return field_key['stringValue']
    raise KeyError(f'Field key {field} not found!')