from zope.interface import Attribute, Interface
def getUser(name):
    """Retrieve the user by the given name.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the user with the given
        name if one exists (or if one is created due to the setting of
        L{IChatService.createUserOnRequest}, or which fails with
        L{twisted.words.ewords.NoSuchUser} if no such user exists.
        """