import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe
def compute_msg(some_serialized_msg):
    return self._encode_parts(some_serialized_msg + [self.not_finished_json], encode_empty=True)