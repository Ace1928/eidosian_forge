from zope.interface import Interface
def getUserNamespaces():
    """
        Report the available user namespaces.

        These are namespaces that contain folders belonging to other users
        access to which this account has been granted.

        @rtype: iterable of two-tuples of strings
        @return: The user namespaces and their hierarchical delimiters. If no
            namespaces of this type exist, None should be returned.
        """