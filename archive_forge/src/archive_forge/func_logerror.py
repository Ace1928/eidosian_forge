import collections.abc
import logging
from typing import Any as TypingAny
from typing import List, Tuple
from pydispatch.dispatcher import (
from pydispatch.robustapply import robustApply
from twisted.internet.defer import Deferred, DeferredList
from twisted.python.failure import Failure
from scrapy.exceptions import StopDownload
from scrapy.utils.defer import maybeDeferred_coro
from scrapy.utils.log import failure_to_exc_info
def logerror(failure: Failure, recv: Any) -> Failure:
    if dont_log is None or not isinstance(failure.value, dont_log):
        logger.error('Error caught on signal handler: %(receiver)s', {'receiver': recv}, exc_info=failure_to_exc_info(failure), extra={'spider': spider})
    return failure