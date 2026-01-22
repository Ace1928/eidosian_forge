import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def get_boto3_client_method_parameters(client, method_name, required=False):
    op = client.meta.method_to_api_mapping.get(method_name)
    input_shape = client._service_model.operation_model(op).input_shape
    if not input_shape:
        parameters = []
    elif required:
        parameters = list(input_shape.required_members)
    else:
        parameters = list(input_shape.members.keys())
    return parameters