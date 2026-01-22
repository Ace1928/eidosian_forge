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
def generate_signed_prekey(self):
    logger.debug('generate_signed_prekey')
    latest_signed_prekey = self.load_latest_signed_prekey(generate=False)
    if latest_signed_prekey is not None:
        if latest_signed_prekey.getId() == self.MAX_SIGNED_PREKEY_ID:
            new_signed_prekey_id = self.MAX_SIGNED_PREKEY_ID / 2 + 1
        else:
            new_signed_prekey_id = latest_signed_prekey.getId() + 1
    else:
        new_signed_prekey_id = 0
    signed_prekey = KeyHelper.generateSignedPreKey(self._identity, new_signed_prekey_id)
    self._store.storeSignedPreKey(signed_prekey.getId(), signed_prekey)
    return signed_prekey