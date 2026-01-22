import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def compute_retry_delay(self, context):
    return self._retry_backoff.delay_amount(context)