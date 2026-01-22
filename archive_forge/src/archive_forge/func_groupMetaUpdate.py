from zope.interface import Attribute, Interface
def groupMetaUpdate(group, meta):
    """
        Callback notifying this user that the metadata for the given
        group has changed.

        @type group: L{IGroup}
        @type meta: C{dict}

        @rtype: L{twisted.internet.defer.Deferred}
        """