import time, calendar
def getquote(self):
    """Get a quote-delimited fragment from self's field."""
    return self.getdelimited('"', '"\r', False)