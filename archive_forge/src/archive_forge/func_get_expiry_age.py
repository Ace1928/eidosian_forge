import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def get_expiry_age(self, **kwargs):
    """Get the number of seconds until the session expires.

        Optionally, this function accepts `modification` and `expiry` keyword
        arguments specifying the modification and expiry of the session.
        """
    try:
        modification = kwargs['modification']
    except KeyError:
        modification = timezone.now()
    try:
        expiry = kwargs['expiry']
    except KeyError:
        expiry = self.get('_session_expiry')
    if not expiry:
        return self.get_session_cookie_age()
    if not isinstance(expiry, (datetime, str)):
        return expiry
    if isinstance(expiry, str):
        expiry = datetime.fromisoformat(expiry)
    delta = expiry - modification
    return delta.days * 86400 + delta.seconds