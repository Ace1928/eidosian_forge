from typing import Dict, Tuple, Union
from twisted.words.protocols.jabber.xmpp_stringprep import (
def userhost(self):
    """
        Extract the bare JID as a unicode string.

        A bare JID does not have a resource part, so this returns either
        C{user@host} or just C{host}.

        @rtype: L{str}
        """
    if self.user:
        return f'{self.user}@{self.host}'
    else:
        return self.host