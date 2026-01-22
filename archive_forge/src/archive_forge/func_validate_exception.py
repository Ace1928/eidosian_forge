from __future__ import annotations
import inspect
import logging
from tenacity import retry, wait_exponential, stop_after_delay, before_sleep_log, retry_unless_exception_type, retry_if_exception_type, retry_if_exception
from typing import Optional, Union, Tuple, Type, TYPE_CHECKING
def validate_exception(self, e: BaseException) -> bool:
    if e.args and e.args[0] == 'PING':
        print('EXCLUDED PING')
        return False
    return isinstance(e, self.exception_types) and (not isinstance(e, self.excluded_types))