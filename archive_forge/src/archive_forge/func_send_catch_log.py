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
def send_catch_log(signal: TypingAny=Any, sender: TypingAny=Anonymous, *arguments: TypingAny, **named: TypingAny) -> List[Tuple[TypingAny, TypingAny]]:
    """Like pydispatcher.robust.sendRobust but it also logs errors and returns
    Failures instead of exceptions.
    """
    dont_log = named.pop('dont_log', ())
    dont_log = tuple(dont_log) if isinstance(dont_log, collections.abc.Sequence) else (dont_log,)
    dont_log += (StopDownload,)
    spider = named.get('spider', None)
    responses: List[Tuple[TypingAny, TypingAny]] = []
    for receiver in liveReceivers(getAllReceivers(sender, signal)):
        result: TypingAny
        try:
            response = robustApply(receiver, *arguments, signal=signal, sender=sender, **named)
            if isinstance(response, Deferred):
                logger.error('Cannot return deferreds from signal handler: %(receiver)s', {'receiver': receiver}, extra={'spider': spider})
        except dont_log:
            result = Failure()
        except Exception:
            result = Failure()
            logger.error('Error caught on signal handler: %(receiver)s', {'receiver': receiver}, exc_info=True, extra={'spider': spider})
        else:
            result = response
        responses.append((receiver, result))
    return responses