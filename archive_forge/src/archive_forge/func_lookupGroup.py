from zope.interface import Attribute, Interface
def lookupGroup(name):
    """Retrieve a group by name.

        Unlike C{getGroup}, this will never implicitly create a group.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the group by the given
        name, or which fails with L{twisted.words.ewords.NoSuchGroup}.
        """