from typing import List
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.task import cooperate
from twisted.python import log
from twisted.python.reflect import safe_str
def startStreaming(self):
    """
        This should be called by the consumer when the producer is registered.

        Start streaming data to the consumer.
        """
    self._coopTask = cooperate(self._pull())