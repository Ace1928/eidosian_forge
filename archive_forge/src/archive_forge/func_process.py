from ..protocol.senderkeydistributionmessage import SenderKeyDistributionMessage
from ..invalidkeyidexception import InvalidKeyIdException
from ..invalidkeyexception import InvalidKeyException
from ..util.keyhelper import KeyHelper
def process(self, senderKeyName, senderKeyDistributionMessage):
    """
        :type senderKeyName: SenderKeyName
        :type senderKeyDistributionMessage: SenderKeyDistributionMessage
        """
    senderKeyRecord = self.senderKeyStore.loadSenderKey(senderKeyName)
    senderKeyRecord.addSenderKeyState(senderKeyDistributionMessage.getId(), senderKeyDistributionMessage.getIteration(), senderKeyDistributionMessage.getChainKey(), senderKeyDistributionMessage.getSignatureKey())
    self.senderKeyStore.storeSenderKey(senderKeyName, senderKeyRecord)