from zope.interface import Attribute, Interface
def positionReceived(latitude, longitude):
    """
        Method called when a position is received.

        @param latitude: The latitude of the received position.
        @type latitude: L{twisted.positioning.base.Coordinate}
        @param longitude: The longitude of the received position.
        @type longitude: L{twisted.positioning.base.Coordinate}
        """