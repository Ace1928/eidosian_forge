import six
import sys
import time
import traceback
import random
import asyncio
import functools
def should_reject(self, attempt):
    reject = False
    if attempt.has_exception:
        reject |= self._retry_on_exception(attempt.value[1])
    else:
        reject |= self._retry_on_result(attempt.value)
    return reject