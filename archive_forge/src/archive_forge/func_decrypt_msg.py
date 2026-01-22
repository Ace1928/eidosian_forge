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
def decrypt_msg(self, senderid, data, unpad):
    logger.debug('decrypt_msg(senderid=%s, data=[omitted], unpad=%s)' % (senderid, unpad))
    msg = WhisperMessage(serialized=data)
    try:
        plaintext = self._get_session_cipher(senderid).decryptMsg(msg)
        return self._unpad(plaintext) if unpad else plaintext
    except NoSessionException:
        raise exceptions.NoSessionException()
    except InvalidKeyIdException:
        raise exceptions.InvalidKeyIdException()
    except InvalidMessageException:
        raise exceptions.InvalidMessageException()
    except DuplicateMessageException:
        raise exceptions.DuplicateMessageException()