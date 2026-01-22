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
def group_decrypt(self, groupid, participantid, data):
    logger.debug('group_decrypt(groupid=%s, participantid=%s, data=[omitted])' % (groupid, participantid))
    group_cipher = self._get_group_cipher(groupid, participantid)
    try:
        plaintext = group_cipher.decrypt(data)
        plaintext = self._unpad(plaintext)
        return plaintext
    except NoSessionException:
        raise exceptions.NoSessionException()
    except DuplicateMessageException:
        raise exceptions.DuplicateMessageException()
    except InvalidMessageException:
        raise exceptions.InvalidMessageException()