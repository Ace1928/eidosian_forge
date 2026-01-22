import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def discovery_operation_kwargs(self, **kwargs):
    input_keys = self.discovery_operation_keys
    if not kwargs.get('Identifiers'):
        kwargs.pop('Operation', None)
        kwargs.pop('Identifiers', None)
    return {k: v for k, v in kwargs.items() if k in input_keys}