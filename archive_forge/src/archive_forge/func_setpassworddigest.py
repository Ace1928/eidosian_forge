from logging import getLogger
from suds import *
from suds.sudsobject import Object
from suds.sax.element import Element
from suds.sax.date import DateTime, UtcTimezone
from datetime import datetime, timedelta
def setpassworddigest(self, passwd_digest):
    """
        Set password digest which is a text returned by
        auth WS.
        """
    self.password_digest = passwd_digest