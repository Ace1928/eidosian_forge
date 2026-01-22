from zope.interface import Attribute, Interface
def disownHandlerParent(parent):
    """
        Remove the parent of the handler.

        @type parent: L{IXMPPHandlerCollection}
        """