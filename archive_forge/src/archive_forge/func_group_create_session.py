from axolotl.util.keyhelper import KeyHelper
from axolotl.identitykeypair import IdentityKeyPair
from axolotl.groups.senderkeyname import SenderKeyName
from axolotl.axolotladdress import AxolotlAddress
from axolotl.sessioncipher import SessionCipher
from axolotl.groups.groupcipher import GroupCipher
from axolotl.groups.groupsessionbuilder import GroupSessionBuilder
from axolotl.sessionbuilder import SessionBuilder
from axolotl.protocol.prekeywhispermessage import PreKeyWhisperMessage
from axolotl.protocol.whispermessage import WhisperMessage
from axolotl.state.prekeybundle import PreKeyBundle
from axolotl.untrustedidentityexception import UntrustedIdentityException
from axolotl.invalidmessageexception import InvalidMessageException
from axolotl.duplicatemessagexception import DuplicateMessageException
from axolotl.invalidkeyidexception import InvalidKeyIdException
from axolotl.nosessionexception import NoSessionException
from axolotl.protocol.senderkeydistributionmessage import SenderKeyDistributionMessage
from axolotl.state.axolotlstore import AxolotlStore
from yowsup.axolotl.store.sqlite.liteaxolotlstore import LiteAxolotlStore
from yowsup.axolotl import exceptions
import random
import logging
import sys
def group_create_session(self, groupid, participantid, skmsgdata):
    """
        :param groupid:
        :type groupid: str
        :param participantid:
        :type participantid: str
        :param skmsgdata:
        :type skmsgdata: bytearray
        :return:
        :rtype:
        """
    logger.debug('group_create_session(groupid=%s, participantid=%s, skmsgdata=[omitted])' % (groupid, participantid))
    senderKeyName = SenderKeyName(groupid, AxolotlAddress(participantid, 0))
    senderkeydistributionmessage = SenderKeyDistributionMessage(serialized=skmsgdata)
    self._group_session_builder.process(senderKeyName, senderkeydistributionmessage)