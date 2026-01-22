from zope.interface import Interface
def getMessageDelivery():
    """
        Return an L{IMessageDelivery} object.

        This will be called once per message.
        """