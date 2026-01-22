from zope.interface import Attribute, Interface
def climbReceived(climb):
    """
        Method called when the climb is received.

        @param climb: The climb of the mobile object.
        @type climb: L{twisted.positioning.base.Climb}
        """