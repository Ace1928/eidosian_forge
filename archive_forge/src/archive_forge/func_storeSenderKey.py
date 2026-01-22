import abc
@abc.abstractmethod
def storeSenderKey(self, senderKeyId, senderKeyRecord):
    """
        :type senderKeyId: str
        :type senderKeyRecord: SenderKeyRecord
        """