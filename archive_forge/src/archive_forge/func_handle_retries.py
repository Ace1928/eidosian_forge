import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def handle_retries(self, request_dict, response, operation, **kwargs):
    if response is None:
        return None
    _, response = response
    status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    error_code = response.get('Error', {}).get('Code')
    if status != 421 and error_code != 'InvalidEndpointException':
        return None
    context = request_dict.get('context', {})
    ids = context.get('discovery', {}).get('identifiers')
    if ids is None:
        return None
    self._manager.delete_endpoints(Operation=operation.name, Identifiers=ids)
    return 0