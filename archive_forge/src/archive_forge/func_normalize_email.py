import unicodedata
import warnings
from django.conf import settings
from django.contrib.auth import password_validation
from django.contrib.auth.hashers import (
from django.db import models
from django.utils.crypto import get_random_string, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _
@classmethod
def normalize_email(cls, email):
    """
        Normalize the email address by lowercasing the domain part of it.
        """
    email = email or ''
    try:
        email_name, domain_part = email.strip().rsplit('@', 1)
    except ValueError:
        pass
    else:
        email = email_name + '@' + domain_part.lower()
    return email