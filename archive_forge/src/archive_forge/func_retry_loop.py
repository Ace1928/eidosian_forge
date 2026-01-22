import ssl
import time
import socket
import logging
from datetime import datetime, timedelta
from functools import wraps
from libcloud.utils.py3 import httplib
from libcloud.common.exceptions import RateLimitReachedError
@wraps(func)
def retry_loop(*args, **kwargs):
    current_delay = self.retry_delay
    end = datetime.now() + timedelta(seconds=self.timeout)
    while True:
        try:
            return transform_ssl_error(func, *args, **kwargs)
        except Exception as exc:
            if isinstance(exc, RateLimitReachedError):
                time.sleep(exc.retry_after)
                current_delay = self.retry_delay
                end = datetime.now() + timedelta(seconds=exc.retry_after + self.timeout)
            elif datetime.now() >= end:
                raise
            elif self.should_retry(exc):
                time.sleep(current_delay)
                current_delay *= self.backoff
            else:
                raise