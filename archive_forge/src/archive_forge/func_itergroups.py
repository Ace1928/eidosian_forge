from zope.interface import Attribute, Interface
def itergroups():
    """Return all groups available on this service.

        @rtype: C{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with a list of C{IGroup} providers.
        """