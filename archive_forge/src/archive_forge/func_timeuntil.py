import datetime
from django.utils.html import avoid_wrapping
from django.utils.timezone import is_aware
from django.utils.translation import gettext, ngettext_lazy
def timeuntil(d, now=None, time_strings=None, depth=2):
    """
    Like timesince, but return a string measuring the time until the given time.
    """
    return timesince(d, now, reversed=True, time_strings=time_strings, depth=depth)