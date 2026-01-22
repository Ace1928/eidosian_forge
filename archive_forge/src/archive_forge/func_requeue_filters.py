from kombu import exceptions as kombu_exc
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.utils import kombu_utils as ku
@property
def requeue_filters(self):
    """List of filters (callbacks) to request a message to be requeued.

        The callback(s) will be activated before the message has been acked and
        it can be used to instruct the dispatcher to requeue the message
        instead of processing it. The callback, when called, will be provided
        two positional parameters; the first being the message data and the
        second being the message object. Using these provided parameters the
        filter should return a truthy object if the message should be requeued
        and a falsey object if it should not.
        """
    return self._requeue_filters