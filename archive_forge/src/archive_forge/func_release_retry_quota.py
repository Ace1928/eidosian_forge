import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def release_retry_quota(self, context, http_response, **kwargs):
    if http_response is None:
        return
    status_code = http_response.status_code
    if 200 <= status_code < 300:
        if 'retry_quota_capacity' not in context:
            self._quota.release(self._NO_RETRY_INCREMENT)
        else:
            capacity_amount = context['retry_quota_capacity']
            self._quota.release(capacity_amount)