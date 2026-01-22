from zope.interface import Attribute, Interface
Create a new user with the given name.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the created user, or
        with fails with L{twisted.words.ewords.DuplicateUser} if a
        user by that name exists already.
        