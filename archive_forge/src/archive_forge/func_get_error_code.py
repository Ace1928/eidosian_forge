import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def get_error_code(self):
    """Check if there was a parsed response with an error code.

        If we could not find any error codes, ``None`` is returned.

        """
    if self.parsed_response is None:
        return
    error = self.parsed_response.get('Error', {})
    if not isinstance(error, dict):
        return
    return error.get('Code')