from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import FtdServerError, HTTPMethod
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError, FILE_MODEL_NAME
def is_download_operation(op_spec):
    return op_spec[OperationField.METHOD] == HTTPMethod.GET and op_spec[OperationField.MODEL_NAME] == FILE_MODEL_NAME