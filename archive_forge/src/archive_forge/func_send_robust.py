import asyncio
import logging
import threading
import weakref
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.utils.inspect import func_accepts_kwargs
def send_robust(self, sender, **named):
    """
        Send signal from sender to all connected receivers catching errors.

        If any receivers are asynchronous, they are called after all the
        synchronous receivers via a single call to async_to_sync(). They are
        also executed concurrently with asyncio.gather().

        Arguments:

            sender
                The sender of the signal. Can be any Python object (normally one
                registered with a connect if you actually want something to
                occur).

            named
                Named arguments which will be passed to receivers.

        Return a list of tuple pairs [(receiver, response), ... ].

        If any receiver raises an error (specifically any subclass of
        Exception), return the error instance as the result for that receiver.
        """
    if not self.receivers or self.sender_receivers_cache.get(sender) is NO_RECEIVERS:
        return []
    responses = []
    sync_receivers, async_receivers = self._live_receivers(sender)
    for receiver in sync_receivers:
        try:
            response = receiver(signal=self, sender=sender, **named)
        except Exception as err:
            self._log_robust_failure(receiver, err)
            responses.append((receiver, err))
        else:
            responses.append((receiver, response))
    if async_receivers:

        async def asend_and_wrap_exception(receiver):
            try:
                response = await receiver(signal=self, sender=sender, **named)
            except Exception as err:
                self._log_robust_failure(receiver, err)
                return err
            return response

        async def asend():
            async_responses = await asyncio.gather(*(asend_and_wrap_exception(receiver) for receiver in async_receivers))
            return zip(async_receivers, async_responses)
        responses.extend(async_to_sync(asend)())
    return responses