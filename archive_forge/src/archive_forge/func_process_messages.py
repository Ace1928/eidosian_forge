import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe
def process_messages(self, obj):
    if isinstance(obj, list) and obj:
        if obj[0] == MessageEncoder.message_key:
            if obj[1]:
                obj[3] = mark_safe(obj[3])
            return Message(*obj[2:])
        return [self.process_messages(item) for item in obj]
    if isinstance(obj, dict):
        return {key: self.process_messages(value) for key, value in obj.items()}
    return obj