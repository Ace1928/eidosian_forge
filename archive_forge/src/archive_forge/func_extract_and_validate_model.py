from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def extract_and_validate_model():
    model = op_name[len(OperationNamePrefix.UPSERT):]
    if not self._conn.get_model_spec(model):
        raise FtdInvalidOperationNameError(op_name)
    return model