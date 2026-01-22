import unicodedata
import warnings
from django.conf import settings
from django.contrib.auth import password_validation
from django.contrib.auth.hashers import (
from django.db import models
from django.utils.crypto import get_random_string, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _
def set_unusable_password(self):
    self.password = make_password(None)