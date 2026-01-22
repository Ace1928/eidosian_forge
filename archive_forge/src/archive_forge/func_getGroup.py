from zope.interface import Attribute, Interface
def getGroup(name, client):
    """
        Get a Group for a client.

        Duplicates L{IAccount.getGroup}.

        @type name: string
        @type client: L{Client<IClient>}

        @rtype: L{Group<IGroup>}
        """