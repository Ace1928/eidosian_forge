import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def get_expire_at_browser_close(self):
    """
        Return ``True`` if the session is set to expire when the browser
        closes, and ``False`` if there's an expiry date. Use
        ``get_expiry_date()`` or ``get_expiry_age()`` to find the actual expiry
        date/age, if there is one.
        """
    if (expiry := self.get('_session_expiry')) is None:
        return settings.SESSION_EXPIRE_AT_BROWSER_CLOSE
    return expiry == 0