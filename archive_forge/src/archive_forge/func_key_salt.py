import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
@property
def key_salt(self):
    return 'django.contrib.sessions.' + self.__class__.__qualname__