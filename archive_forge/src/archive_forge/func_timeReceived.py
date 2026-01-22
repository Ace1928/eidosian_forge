from zope.interface import Attribute, Interface
def timeReceived(time):
    """
        Method called when time and date information arrives.

        @param time: The date and time (expressed in UTC unless otherwise
            specified).
        @type time: L{datetime.datetime}
        """