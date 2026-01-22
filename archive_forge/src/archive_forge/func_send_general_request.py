from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def send_general_request(self, operation_name, params):

    def stop_if_check_mode():
        if self._check_mode:
            raise CheckModeException()
    self.validate_params(operation_name, params)
    stop_if_check_mode()
    data, query_params, path_params = _get_user_params(params)
    op_spec = self.get_operation_spec(operation_name)
    url, method = (op_spec[OperationField.URL], op_spec[OperationField.METHOD])
    return self._send_request(url, method, data, path_params, query_params)