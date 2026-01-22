from axolotl.state.axolotlstore import AxolotlStore
from .liteidentitykeystore import LiteIdentityKeyStore
from .liteprekeystore import LitePreKeyStore
from .litesessionstore import LiteSessionStore
from .litesignedprekeystore import LiteSignedPreKeyStore
from .litesenderkeystore import LiteSenderKeyStore
import sqlite3
def loadPreKeys(self):
    return self.preKeyStore.loadPendingPreKeys()