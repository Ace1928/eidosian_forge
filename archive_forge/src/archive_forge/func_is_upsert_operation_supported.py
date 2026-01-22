from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
@classmethod
def is_upsert_operation_supported(cls, operations):
    """
        Checks if all operations required for upsert object operation are defined in 'operations'.

        :param operations: specification of the operations supported by model
        :type operations: dict
        :return: True if all criteria required to provide requested called operation are satisfied, otherwise False
        :rtype: bool
        """
    has_edit_op = next((name for name, spec in iteritems(operations) if cls.is_edit_operation(name, spec)), None)
    has_get_list_op = next((name for name, spec in iteritems(operations) if cls.is_get_list_operation(name, spec)), None)
    return has_edit_op and has_get_list_op