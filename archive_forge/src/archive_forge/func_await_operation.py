from __future__ import absolute_import, division, print_function
import time
import math
from ansible_collections.purestorage.fusion.plugins.module_utils.errors import (
def await_operation(fusion, operation, fail_playbook_if_operation_fails=True):
    """
    Waits for given operation to finish.
    Throws an exception by default if the operation fails.
    """
    op_api = purefusion.OperationsApi(fusion)
    operation_get = None
    while True:
        try:
            operation_get = op_api.get_operation(operation.id)
            if operation_get.status == 'Succeeded':
                return operation_get
            if operation_get.status == 'Failed':
                if fail_playbook_if_operation_fails:
                    raise OperationException(operation_get)
                return operation_get
        except HTTPError as err:
            raise OperationException(operation, http_error=err)
        time.sleep(int(math.ceil(operation_get.retry_in / 1000)))