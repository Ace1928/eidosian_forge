import smtplib
import ssl
import threading
from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend
from django.core.mail.message import sanitize_address
from django.core.mail.utils import DNS_NAME
from django.utils.functional import cached_property
@property
def connection_class(self):
    return smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP