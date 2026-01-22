from typing import Dict, Tuple, Union
from twisted.words.protocols.jabber.xmpp_stringprep import (
def prep(user: Union[str, None], host: str, resource: Union[str, None]) -> Tuple[Union[str, None], str, Union[str, None]]:
    """
    Perform stringprep on all JID fragments.

    @param user: The user part of the JID.
    @type user: L{str}
    @param host: The host part of the JID.
    @type host: L{str}
    @param resource: The resource part of the JID.
    @type resource: L{str}
    @return: The given parts with stringprep applied.
    @rtype: L{tuple}
    """
    if user:
        try:
            user = nodeprep.prepare(str(user))
        except UnicodeError:
            raise InvalidFormat('Invalid character in username')
    else:
        user = None
    if not host:
        raise InvalidFormat('Server address required.')
    else:
        try:
            host = nameprep.prepare(str(host))
        except UnicodeError:
            raise InvalidFormat('Invalid character in hostname')
    if resource:
        try:
            resource = resourceprep.prepare(str(resource))
        except UnicodeError:
            raise InvalidFormat('Invalid character in resource')
    else:
        resource = None
    return (user, host, resource)