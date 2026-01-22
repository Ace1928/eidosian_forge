from zope.interface import Attribute, Interface
def positionErrorReceived(positionError):
    """
        Method called when position error is received.

        @param positionError: The position error.
        @type positionError: L{twisted.positioning.base.PositionError}
        """