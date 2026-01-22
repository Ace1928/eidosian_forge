import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_allowed_domains(self, allowed_domains):
    """Set the sequence of allowed domains, or None."""
    if allowed_domains is not None:
        allowed_domains = tuple(allowed_domains)
    self._allowed_domains = allowed_domains