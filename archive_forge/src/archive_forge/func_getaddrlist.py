import time, calendar
def getaddrlist(self):
    """Parse all addresses.

        Returns a list containing all of the addresses.
        """
    result = []
    while self.pos < len(self.field):
        ad = self.getaddress()
        if ad:
            result += ad
        else:
            result.append(('', ''))
    return result